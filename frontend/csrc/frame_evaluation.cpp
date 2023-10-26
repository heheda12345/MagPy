#define PY_SSIZE_T_CLEAN
#include "csrc.h"
#include <Python.h>
#include <cellobject.h>
#include <frameobject.h>
#include <map>
#include <object.h>
#include <pythread.h>
#include <sstream>
#include <string>
#include <vector>

#define unlikely(x) __builtin_expect((x), 0)

#define CHECK(cond)                                                            \
    if (unlikely(!(cond))) {                                                   \
        fprintf(stderr, "DEBUG CHECK FAILED: %s:%d\n", __FILE__, __LINE__);    \
        abort();                                                               \
    } else {                                                                   \
    }

#define NULL_CHECK(val)                                                        \
    if (unlikely((val) == NULL)) {                                             \
        fprintf(stderr, "NULL ERROR: %s:%d\n", __FILE__, __LINE__);            \
        PyErr_Print();                                                         \
        abort();                                                               \
    } else {                                                                   \
    }

#define TO_PyBool(val) ((val) ? Py_True : Py_False)
#define PRINT_PYERR                                                            \
    {                                                                          \
        if (PyErr_Occurred())                                                  \
            PyErr_Print();                                                     \
    }

static PyObject *skip_files = Py_None;
static Py_tss_t eval_frame_callback_key = Py_tss_NEEDS_INIT;
static int active_working_threads = 0;
static PyObject *(*previous_eval_frame)(PyThreadState *tstate,
                                        PyFrameObject *frame,
                                        int throw_flag) = NULL;
static size_t cache_entry_extra_index = -1;
static std::vector<int *> frame_id_list;
static void ignored(void *obj) {}

frontend_csrc::ProgramCache program_cache;
static int frame_count = 0;

bool need_postprocess = false;
static std::map<size_t, PyObject *> frame_id_to_code_map;

static void pylog(std::string message, const char *level = "info") {
    static PyObject *pModule;
    static PyObject *pFuncInfo;
    if (pModule == NULL) {
        pModule = PyImport_ImportModule("logging");
        pFuncInfo = PyObject_GetAttrString(pModule, level);
        CHECK(pFuncInfo != NULL)
        CHECK(PyCallable_Check(pFuncInfo))
    }
    PyObject *pArgs = PyTuple_Pack(1, PyUnicode_FromString(message.c_str()));
    PyObject_CallObject(pFuncInfo, pArgs);
    Py_XDECREF(pArgs);
}

inline static PyObject *get_current_eval_frame_callback() {
    void *result = PyThread_tss_get(&eval_frame_callback_key);
    if (unlikely(result == NULL)) {
        return (PyObject *)Py_None;
    } else {
        return (PyObject *)result;
    }
}

inline static void set_eval_frame_callback(PyObject *obj) {
    PyThread_tss_set(&eval_frame_callback_key, obj);
}

inline static int get_frame_id(PyCodeObject *code) {
    int *frame_id;
    _PyCode_GetExtra((PyObject *)code, cache_entry_extra_index,
                     (void **)&frame_id);
    if (frame_id == NULL) {
        // WARNING: memory leak here
        frame_id = new int(frame_count++);
        _PyCode_SetExtra((PyObject *)code, cache_entry_extra_index, frame_id);
        frame_id_list.push_back(frame_id);
    }
    return *frame_id;
}

inline static PyObject *eval_frame_default(PyThreadState *tstate,
                                           PyFrameObject *frame,
                                           int throw_flag) {
    if (tstate == NULL) {
        tstate = PyThreadState_GET();
    }
    if (previous_eval_frame) {
        return previous_eval_frame(tstate, frame, throw_flag);
    } else {
        return _PyEval_EvalFrameDefault(tstate, frame, throw_flag);
    }
}

inline static PyObject *eval_custom_code(PyThreadState *tstate,
                                         PyFrameObject *frame,
                                         PyCodeObject *code, PyObject *code_map,
                                         int throw_flag, bool trace_bytecode,
                                         PyObject *trace_func) {
    Py_ssize_t ncells = 0;
    Py_ssize_t nfrees = 0;
    Py_ssize_t nlocals_new = code->co_nlocals;
    Py_ssize_t nlocals_old = frame->f_code->co_nlocals;

    ncells = PyTuple_GET_SIZE(code->co_cellvars);
    nfrees = PyTuple_GET_SIZE(code->co_freevars);

    NULL_CHECK(tstate);
    NULL_CHECK(frame);
    NULL_CHECK(code);
    CHECK(nlocals_new >= nlocals_old);
    CHECK(ncells == PyTuple_GET_SIZE(frame->f_code->co_cellvars));
    CHECK(nfrees == PyTuple_GET_SIZE(frame->f_code->co_freevars));

    PyFrameObject *shadow = PyFrame_New(tstate, code, frame->f_globals, NULL);
    if (shadow == NULL) {
        return NULL;
    }

    shadow->f_trace_opcodes = trace_bytecode;
    Py_CLEAR(shadow->f_trace);
    Py_XINCREF(trace_func);
    shadow->f_trace = trace_func;
    PyObject **fastlocals_old = frame->f_localsplus;
    PyObject **fastlocals_new = shadow->f_localsplus;

    for (Py_ssize_t i = 0; i < nlocals_old; i++) {
        Py_XINCREF(fastlocals_old[i]);
        fastlocals_new[i] = fastlocals_old[i];
    }

    for (Py_ssize_t i = 0; i < ncells + nfrees; i++) {
        Py_XINCREF(fastlocals_old[nlocals_old + i]);
        fastlocals_new[nlocals_new + i] = fastlocals_old[nlocals_old + i];
    }

    frame_id_to_code_map[(size_t)shadow] = code_map;
    Py_INCREF(code_map);
    PyObject *result = eval_frame_default(tstate, shadow, throw_flag);
    frame_id_to_code_map.erase((size_t)shadow);
    Py_DECREF(code_map);
    Py_DECREF(shadow);
    return result;
}

// run the callback
static PyObject *_custom_eval_frame(PyThreadState *tstate,
                                    PyFrameObject *_frame, int throw_flag,
                                    PyObject *callback) {
    set_eval_frame_callback(Py_None);
    int frame_id = get_frame_id(_frame->f_code);
    if (frame_id >= program_cache.size()) {
        CHECK(frame_id == program_cache.size());
        frontend_csrc::FrameCache empty;
        empty.push_back(nullptr);
        program_cache.push_back(empty);
    }
    Py_INCREF(_frame);
    PyObject *preprocess = PyTuple_GetItem(callback, 0);
    PyObject *postprocess = PyTuple_GetItem(callback, 1);
    Py_INCREF(preprocess);
    Py_INCREF(postprocess);
    PyObject *result_preprocess =
        PyObject_CallFunction(preprocess, "Oi", _frame, frame_id);
    PyObject *new_code = PyTuple_GetItem(result_preprocess, 0);
    PyObject *trace_func = PyTuple_GetItem(result_preprocess, 1);
    PyObject *code_map = PyTuple_GetItem(result_preprocess, 2);
    Py_INCREF(new_code);
    Py_INCREF(trace_func);
    need_postprocess = false;
    PyObject *result =
        eval_custom_code(tstate, _frame, (PyCodeObject *)new_code, code_map,
                         false, true, trace_func);
    // _frame->
    // PyObject *result = _PyEval_EvalFrameDefault(tstate, _frame, throw_flag);
    /*
    _frame->f_trace = NULL;
    */
    if (need_postprocess) {
        PyObject *result_postprocess =
            PyObject_CallFunction(postprocess, "O", (PyObject *)_frame);
    }
    Py_DECREF(_frame);
    Py_DECREF(preprocess);
    Py_DECREF(postprocess);
    Py_DECREF(trace_func);

    // set_eval_frame_callback(callback);
    return result;
}

// run the callback or the default
static PyObject *custom_eval_frame_shim(PyThreadState *tstate,
                                        PyFrameObject *frame, int throw_flag) {
    PyObject *callback = get_current_eval_frame_callback();

    if (callback == Py_None) {
        return _PyEval_EvalFrameDefault(tstate, frame, throw_flag);
    }
    assert(PyObject_IsInstance(skip_files, (PyObject *)&PySet_Type));
    if (PySet_Contains(skip_files, frame->f_code->co_filename)) {
        return _PyEval_EvalFrameDefault(tstate, frame, throw_flag);
    }
    PyObject *result = _custom_eval_frame(tstate, frame, throw_flag, callback);
    return result;
}

inline static void enable_eval_frame_shim(PyThreadState *tstate) {
    if (_PyInterpreterState_GetEvalFrameFunc(tstate->interp) !=
        &custom_eval_frame_shim) {
        previous_eval_frame =
            _PyInterpreterState_GetEvalFrameFunc(tstate->interp);
        _PyInterpreterState_SetEvalFrameFunc(tstate->interp,
                                             &custom_eval_frame_shim);
    }
}

inline static void enable_eval_frame_default(PyThreadState *tstate) {
    if (_PyInterpreterState_GetEvalFrameFunc(tstate->interp) !=
        previous_eval_frame) {
        _PyInterpreterState_SetEvalFrameFunc(tstate->interp,
                                             previous_eval_frame);
        previous_eval_frame = NULL;
    }
}

static PyObject *increse_working_threads(PyThreadState *tstate) {
    enable_eval_frame_shim(tstate);
    Py_RETURN_NONE;
}

static PyObject *decrese_working_threads(PyThreadState *tstate) {
    enable_eval_frame_default(tstate);
    Py_RETURN_NONE;
}

static PyObject *set_eval_frame(PyObject *self, PyObject *args) {
    PyObject *new_callback = NULL;
    if (!PyArg_ParseTuple(args, "O", &new_callback)) {
        PRINT_PYERR;
        PyErr_SetString(PyExc_TypeError, "invalid parameter");
        return NULL;
    }
    if (new_callback != Py_None) {
        if (!PyTuple_Check(new_callback) || PyTuple_Size(new_callback) != 2 ||
            PyCallable_Check(PyTuple_GetItem(new_callback, 0)) != 1 ||
            PyCallable_Check(PyTuple_GetItem(new_callback, 1)) != 1) {
            PyErr_SetString(PyExc_TypeError, "should be callables");
            return NULL;
        }
    }
    PyThreadState *tstate = PyThreadState_GET();
    PyObject *old_callback = get_current_eval_frame_callback();
    Py_INCREF(old_callback);

    if (old_callback != Py_None && new_callback == Py_None) {
        decrese_working_threads(tstate);
    } else if (old_callback == Py_None && new_callback != Py_None) {
        increse_working_threads(tstate);
    }

    Py_INCREF(new_callback);
    Py_DECREF(old_callback);

    set_eval_frame_callback(new_callback);
    return old_callback;
}

// TODO: in a more elegant way
static PyObject *set_skip_files(PyObject *self, PyObject *args) {
    if (skip_files != Py_None) {
        Py_DECREF(skip_files);
    }
    if (!PyArg_ParseTuple(args, "O", &skip_files)) {
        PRINT_PYERR
        PyErr_SetString(PyExc_TypeError, "invalid parameter in set_skip_files");
    }
    Py_INCREF(skip_files);
    Py_RETURN_NONE;
}

static PyObject *set_null_object(PyObject *self, PyObject *args) {
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        PRINT_PYERR
        PyErr_SetString(PyExc_TypeError,
                        "invalid parameter in set_null_object");
    }
    Py_INCREF(obj);
    frontend_csrc::NullObjectSingleton::getInstance().setNullObject(obj);
    Py_RETURN_NONE;
}

static PyObject *get_value_stack_from_top(PyObject *self, PyObject *args) {
    PyFrameObject *frame = NULL;
    int index = 0;
    if (!PyArg_ParseTuple(args, "Oi", &frame, &index)) {
        PRINT_PYERR;
        PyErr_SetString(PyExc_TypeError,
                        "invalid parameter in get_value_stack_from_top");
        return NULL;
    }
    PyObject *value = frame->f_stacktop[-index - 1];
    if (value == NULL) {
        value =
            frontend_csrc::NullObjectSingleton::getInstance().getNullObject();
    }
    Py_INCREF(value);
    return value;
}

static PyObject *get_value_stack_size(PyObject *self, PyObject *args) {
    PyFrameObject *frame = NULL;
    if (!PyArg_ParseTuple(args, "O", &frame)) {
        PRINT_PYERR;
        PyErr_SetString(PyExc_TypeError,
                        "invalid parameter in get_value_stack_size");
        return NULL;
    }
    return PyLong_FromLong((int)(frame->f_stacktop - frame->f_valuestack));
}

static PyObject *add_to_cache(PyObject *self, PyObject *args) {
    int frame_id, callsite_id, id_in_callsite;
    PyObject *check_fn, *graph_fn;
    if (!PyArg_ParseTuple(args, "iiiOO", &frame_id, &callsite_id,
                          &id_in_callsite, &check_fn, &graph_fn)) {
        PRINT_PYERR;
        PyErr_SetString(PyExc_TypeError, "invalid parameter in add_to_cache");
        return NULL;
    }
    if (callsite_id >= program_cache[frame_id].size()) {
        CHECK(callsite_id == program_cache[frame_id].size());
        program_cache[frame_id].push_back(nullptr);
    }
    Py_INCREF(check_fn);
    Py_INCREF(graph_fn);
    frontend_csrc::Cache *entry = new frontend_csrc::Cache{
        check_fn, PyTuple_Pack(2, PyLong_FromLong(id_in_callsite), graph_fn),
        program_cache[frame_id][callsite_id]};
    program_cache[frame_id][callsite_id] = entry;
    Py_RETURN_NONE;
}

static PyObject *guard_match(PyObject *self, PyObject *args) {
    int frame_id, callsite_id;
    PyObject *locals;
    if (!PyArg_ParseTuple(args, "iiO", &frame_id, &callsite_id, &locals)) {
        PRINT_PYERR;
        PyErr_SetString(PyExc_TypeError, "invalid parameter in guard_match");
        return NULL;
    }
    for (frontend_csrc::Cache *entry = program_cache[frame_id][callsite_id];
         entry != NULL; entry = entry->next) {
        PyObject *valid = PyObject_CallOneArg(entry->check_fn, locals);
        if (valid == Py_True) {
            Py_DECREF(valid);
#ifdef LOG_CACHE
            std::stringstream ss;
            ss << "\033[31mguard cache hit: frame_id " << frame_id
               << " callsite_id " << callsite_id << "\033[0m";
            pylog(ss.str());
#endif
            Py_INCREF(entry->graph_fn);
            return entry->graph_fn;
        }
        Py_DECREF(valid);
    }
#ifdef LOG_CACHE
    std::stringstream ss;
    ss << "\033[31mguard cache miss: frame_id " << frame_id << " callsite_id "
       << callsite_id << "\033[0m";
    pylog(ss.str());
#endif
    return PyTuple_Pack(2, PyLong_FromLong(-1), Py_None);
}

static PyObject *reset(PyObject *self, PyObject *args) {
    // as we cannot recover the frame_id assigned by _PyCode_GetExtra, we only
    // clear the cache of each frame, and keeps the program_cache vector
    for (frontend_csrc::FrameCache &frame_cache : program_cache) {
        for (frontend_csrc::Cache *entry : frame_cache) {
            if (entry == nullptr) {
                continue;
            }
            Py_DECREF(entry->check_fn);
            Py_DECREF(entry->graph_fn);
            delete entry;
        }
        frame_cache.clear();
        frame_cache.push_back(nullptr);
    }
    Py_RETURN_NONE;
}

static PyObject *enter_nested_tracer(PyObject *self, PyObject *args) {
    PyThreadState *tstate = PyThreadState_GET();
    tstate->tracing--;
    Py_RETURN_NONE;
}

static PyObject *exit_nested_tracer(PyObject *self, PyObject *args) {
    PyThreadState *tstate = PyThreadState_GET();
    tstate->tracing++;
    Py_RETURN_NONE;
}

static PyObject *stack_effect_py(PyObject *self, PyObject *args) {
    int opcode, oparg, jump;
    PyObject *jump_obj;
    if (!PyArg_ParseTuple(args, "iiO", &opcode, &oparg, &jump_obj)) {
        PRINT_PYERR;
        PyErr_SetString(PyExc_TypeError, "invalid parameter in stack_effect");
        return NULL;
    }
    if (jump_obj == Py_None) {
        jump = -1;
    } else if (jump_obj == Py_True) {
        jump = 1;
    } else if (jump_obj == Py_False) {
        jump = 0;
    }
    frontend_csrc::StackEffect effect =
        frontend_csrc::stack_effect(opcode, oparg, jump);
    return Py_BuildValue("iiiOO", effect.read, effect.write_old,
                         effect.write_new, TO_PyBool(effect.local_effect),
                         TO_PyBool(effect.global_effect));
}

static PyObject *get_code_map(PyObject *self, PyObject *args) {
    PyFrameObject *frame = NULL;
    if (!PyArg_ParseTuple(args, "O", &frame)) {
        PRINT_PYERR;
        PyErr_SetString(PyExc_TypeError, "invalid parameter in get_code_map");
        return NULL;
    }
    PyObject *code_map = frame_id_to_code_map[(size_t)frame];
    Py_INCREF(code_map);
    return code_map;
}

static PyObject *is_bound_method(PyObject *self, PyObject *args) {
    PyObject *obj;
    PyObject *name;
    if (!PyArg_ParseTuple(args, "OO", &obj, &name)) {
        PRINT_PYERR;
        PyErr_SetString(PyExc_TypeError, "invalid parameter in get_method");
        return NULL;
    }
    PyObject *meth = NULL;
    int meth_found = _PyObject_GetMethod(obj, name, &meth);
    if (meth_found) {
        return Py_True;
    } else {
        return Py_False;
    }
}

static PyObject *get_from_freevars(PyObject *self, PyObject *args) {
    PyObject *frame;
    int index;
    if (!PyArg_ParseTuple(args, "Oi", &frame, &index)) {
        PRINT_PYERR;
        PyErr_SetString(PyExc_TypeError,
                        "invalid parameter in get_from_freevars");
        return NULL;
    }
    PyFrameObject *f = (PyFrameObject *)frame;
    PyObject *value = f->f_localsplus[index + f->f_code->co_nlocals];
    if (value == NULL) {
        value =
            frontend_csrc::NullObjectSingleton::getInstance().getNullObject();
    }
    Py_INCREF(value);
    return value;
}

static PyMethodDef _methods[] = {
    {"set_eval_frame", set_eval_frame, METH_VARARGS, NULL},
    {"set_skip_files", set_skip_files, METH_VARARGS, NULL},
    {"set_null_object", set_null_object, METH_VARARGS, NULL},
    {"get_value_stack_from_top", get_value_stack_from_top, METH_VARARGS, NULL},
    {"get_value_stack_size", get_value_stack_size, METH_VARARGS, NULL},
    {"guard_match", guard_match, METH_VARARGS, NULL},
    {"add_to_cache", add_to_cache, METH_VARARGS, NULL},
    {"enter_nested_tracer", enter_nested_tracer, METH_VARARGS, NULL},
    {"exit_nested_tracer", exit_nested_tracer, METH_VARARGS, NULL},
    {"c_reset", reset, METH_VARARGS, NULL},
    {"stack_effect", stack_effect_py, METH_VARARGS, NULL},
    {"mark_need_postprocess",
     [](PyObject *self, PyObject *args) {
         need_postprocess = true;
         Py_RETURN_NONE;
     },
     METH_VARARGS, NULL},
    {"get_next_frame_id",
     [](PyObject *self, PyObject *args) {
         return PyLong_FromLong(frame_count);
     },
     METH_VARARGS, NULL},
    {"get_code_map", get_code_map, METH_VARARGS, NULL},
    {"is_bound_method", is_bound_method, METH_VARARGS, NULL},
    {"get_from_freevars", get_from_freevars, METH_VARARGS, NULL},
    {"parse_rangeiterobject", frontend_csrc::parse_rangeiterobject,
     METH_VARARGS, NULL},
    {"make_rangeiterobject", frontend_csrc::make_rangeiterobject, METH_VARARGS,
     NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT, "frontend.c_api",
    "Module containing hooks to override eval_frame", -1, _methods};

PyMODINIT_FUNC PyInit_c_api(void) {
    cache_entry_extra_index = _PyEval_RequestCodeExtraIndex(ignored);
    if (cache_entry_extra_index < 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "c_api: unable to register cache_entry extra index");
        return NULL;
    }
    int result = PyThread_tss_create(&eval_frame_callback_key);
    CHECK(result == 0);
    Py_INCREF(Py_None);
    set_eval_frame_callback(Py_None);

    PyObject *m = PyModule_Create(&_module);
    return m;
}