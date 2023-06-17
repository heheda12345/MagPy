#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cache.h>
#include <frameobject.h>
#include <pythread.h>

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

static PyObject *skip_files = Py_None;
static Py_tss_t eval_frame_callback_key = Py_tss_NEEDS_INIT;
static int active_working_threads = 0;
static PyObject *(*previous_eval_frame)(PyThreadState *tstate,
                                        PyFrameObject *frame,
                                        int throw_flag) = NULL;
static size_t cache_entry_extra_index = -1;
static void ignored(void *obj) {}

ProgramCache program_cache;
static int frame_count = 0;

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
                                         PyCodeObject *code, int throw_flag) {
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

    PyObject *result = eval_frame_default(tstate, shadow, throw_flag);

    Py_DECREF(shadow);
    return result;
}

// run the callback
static PyObject *_custom_eval_frame(PyThreadState *tstate,
                                    PyFrameObject *_frame, int throw_flag,
                                    PyObject *callback) {
    set_eval_frame_callback(Py_None);
    int frame_id = get_frame_id(_frame->f_code);
    printf("frame_id %d\n", frame_id);
    if (frame_id >= program_cache.size()) {
        CHECK(frame_id == program_cache.size());
        FrameCache empty;
        empty.push_back(nullptr);
        program_cache.push_back(empty);
    }
    Py_INCREF(_frame);
    PyObject *preprocess = PyTuple_GetItem(callback, 0);
    PyObject *postprocess = PyTuple_GetItem(callback, 1);
    PyObject *trace_func = PyTuple_GetItem(callback, 2);
    PyObject *frame_id_object = PyLong_FromLong(frame_id);
    PyObject *result_preprocess =
        PyObject_CallFunction(preprocess, "Oi", _frame, frame_id_object);
    PyObject *new_code = PyTuple_GetItem(result_preprocess, 0);
    PyObject *check_fn = PyTuple_GetItem(result_preprocess, 1);
    PyObject *graph_fn = PyTuple_GetItem(result_preprocess, 2);
    if (program_cache[frame_id][0] == nullptr) {
        program_cache[frame_id][0] = new Cache{check_fn, graph_fn, nullptr};
    }
    Py_DecRef(frame_id_object);
    PyObject *result =
        eval_custom_code(tstate, _frame, (PyCodeObject *)new_code, false);
    /*
    _frame->f_trace = trace_func;
    _frame->f_trace_opcodes = 1;
    PyObject* result = _PyEval_EvalFrameDefault(tstate, _frame, throw_flag);
    _frame->f_trace = NULL;
    */
    PyObject *result_postprocess =
        PyObject_CallFunction(postprocess, "O", (PyObject *)_frame);
    Py_DECREF(_frame);

    set_eval_frame_callback(callback);
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
    return _custom_eval_frame(tstate, frame, throw_flag, callback);
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
    active_working_threads = active_working_threads + 1;
    if (active_working_threads > 0) {
        enable_eval_frame_shim(tstate);
    }
    Py_RETURN_NONE;
}

static PyObject *decrese_working_threads(PyThreadState *tstate) {
    if (active_working_threads > 0) {
        active_working_threads = active_working_threads - 1;
        if (active_working_threads == 0) {
            enable_eval_frame_default(tstate);
        }
    }
    Py_RETURN_NONE;
}

static PyObject *set_eval_frame(PyObject *self, PyObject *args) {
    PyObject *new_callback = NULL;
    if (!PyArg_ParseTuple(args, "O", &new_callback)) {
        PyErr_SetString(PyExc_TypeError, "invalid parameter");
        return NULL;
    }
    if (new_callback != Py_None) {
        if (!PyTuple_Check(new_callback) || PyTuple_Size(new_callback) != 3 ||
            PyCallable_Check(PyTuple_GetItem(new_callback, 0)) != 1 ||
            PyCallable_Check(PyTuple_GetItem(new_callback, 1)) != 1 ||
            PyCallable_Check(PyTuple_GetItem(new_callback, 2)) != 1) {
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
        PyErr_SetString(PyExc_TypeError, "invalid parameter in set_skip_files");
    }
    Py_INCREF(skip_files);
    Py_RETURN_NONE;
}

static PyObject *get_value_stack_from_top(PyObject *self, PyObject *args) {
    PyFrameObject *frame = NULL;
    int index = 0;
    if (!PyArg_ParseTuple(args, "Oi", &frame, &index)) {
        PyErr_SetString(PyExc_TypeError,
                        "invalid parameter in get_value_stack_from_top");
        return NULL;
    }
    PyObject *value = frame->f_stacktop[-index - 1];
    Py_INCREF(value);
    return value;
}

static PyObject *guard_match(PyObject *self, PyObject *args) {
    int frame_id, callsite_id;
    PyObject *locals;
    if (!PyArg_ParseTuple(args, "iiO", &frame_id, &callsite_id, &locals)) {
        PyErr_SetString(PyExc_TypeError, "invalid parameter in guard_match");
        return NULL;
    }
    printf("start run\n");
    for (Cache *entry = program_cache[frame_id][callsite_id]; entry != NULL;
         entry = entry->next) {
        PyObject *valid = PyObject_CallOneArg(entry->check_fn, locals);
        Py_DECREF(valid);
        if (valid == Py_True) {
            printf("guard match\n");
            return entry->graph_fn;
        }
    }
    printf("guard cache miss\n");
    return Py_None;
}

static PyMethodDef _methods[] = {
    {"set_eval_frame", set_eval_frame, METH_VARARGS, NULL},
    {"set_skip_files", set_skip_files, METH_VARARGS, NULL},
    {"get_value_stack_from_top", get_value_stack_from_top, METH_VARARGS, NULL},
    {"guard_match", guard_match, METH_VARARGS, NULL},
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
    return PyModule_Create(&_module);
}