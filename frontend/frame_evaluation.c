#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <pythread.h>
#include <frameobject.h>

#define unlikely(x) __builtin_expect((x), 0)


#define CHECK(cond)                                                         \
    if (unlikely(!(cond))) {                                                \
        fprintf(stderr, "DEBUG CHECK FAILED: %s:%d\n", __FILE__, __LINE__); \
        abort();                                                            \
    } else {                                                                \
    }

static PyObject *skip_files = NULL;
static Py_tss_t eval_frame_callback_key = Py_tss_NEEDS_INIT;
static int active_working_threads = 0;
static PyObject *(*previous_eval_frame)(PyThreadState *tstate,
                                        PyFrameObject* frame, int throw_flag) = NULL;


inline static PyObject* get_current_eval_frame_callback() {
    void* result = PyThread_tss_get(&eval_frame_callback_key);
    if (unlikely(result == NULL)) {
        return (PyObject*)Py_None;
    } else {
        return (PyObject*)result;
    }
}

inline static void set_eval_frame_callback(PyObject* obj) {
    PyThread_tss_set(&eval_frame_callback_key, obj);
}

// run the callback
static PyObject* _custom_eval_frame(
        PyThreadState* tstate,
        PyFrameObject* _frame,
        int throw_flag,
        PyObject* callback){
    set_eval_frame_callback(Py_None);
    PyObject* frame = Py_NewRef(_frame);
    PyObject* result = PyObject_CallFunction(callback, "O", (PyObject*) frame);
    Py_DECREF(frame);
    set_eval_frame_callback(callback);
    return result;
}

// run the callback or the default
static PyObject* custom_eval_frame_shim(
        PyThreadState* tstate,
        PyFrameObject* frame,
        int throw_flag) {
    PyObject* callback = get_current_eval_frame_callback();

    if (callback == Py_None) {
        return _PyEval_EvalFrameDefault(tstate, frame, throw_flag);
    }
    printf("co_filename %s\n", _PyUnicode_AsString(frame->f_code->co_filename));
    if(PySet_Contains(skip_files, frame->f_code->co_filename)) {
        return _PyEval_EvalFrameDefault(tstate, frame, throw_flag);
    }
    return _custom_eval_frame(tstate, frame, throw_flag, callback);
}

inline static void enable_eval_frame_shim(PyThreadState* tstate) {
    if (_PyInterpreterState_GetEvalFrameFunc(tstate->interp) != &custom_eval_frame_shim) {
        previous_eval_frame = _PyInterpreterState_GetEvalFrameFunc(tstate->interp);
        _PyInterpreterState_SetEvalFrameFunc(tstate->interp, &custom_eval_frame_shim);
    }
}


inline static void enable_eval_frame_default(PyThreadState* tstate) {
    if (_PyInterpreterState_GetEvalFrameFunc(tstate->interp) != previous_eval_frame) {
        _PyInterpreterState_SetEvalFrameFunc(tstate->interp, previous_eval_frame);
        previous_eval_frame = NULL;
    }
}

static PyObject* increse_working_threads(PyThreadState* tstate) {
    active_working_threads = active_working_threads + 1;
    if (active_working_threads > 0) {
        enable_eval_frame_shim(tstate);
    }
    Py_RETURN_NONE;
}

static PyObject* decrese_working_threads(PyThreadState* tstate) {
    if (active_working_threads > 0) {
        active_working_threads = active_working_threads - 1;
        if (active_working_threads == 0) {
            enable_eval_frame_default(tstate);
        }
    }
    Py_RETURN_NONE;
}


static PyObject* set_eval_frame(PyObject* self, PyObject* args) {
    PyObject* new_callback = NULL;
    if (!PyArg_ParseTuple(args, "O", &new_callback)) {
        PyErr_SetString(PyExc_TypeError, "invalid parameter");
        return NULL;
    }
    if (new_callback != Py_None && !PyCallable_Check(new_callback)) {
        PyErr_SetString(PyExc_TypeError, "parameter must be callable");
        return NULL;
    }
    PyThreadState* tstate = PyThreadState_GET();
    PyObject* old_callback = get_current_eval_frame_callback();
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

static PyObject* set_skip_files(PyObject* self, PyObject* args) {
    if (!PyArg_ParseTuple(args, "O", &skip_files)) {
        PyErr_SetString(PyExc_TypeError, "invalid parameter");
    }
    Py_RETURN_NONE;
}

static PyMethodDef _methods[] = {
    {"set_eval_frame", set_eval_frame, METH_VARARGS, NULL},
    {"set_skip_files", set_skip_files, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "frontend.c_api",
    "Module containing hooks to override eval_frame",
    -1,
    _methods};

PyMODINIT_FUNC PyInit_c_api(void) {
    int result = PyThread_tss_create(&eval_frame_callback_key);
    CHECK(result == 0);
    Py_INCREF(Py_None);
    set_eval_frame_callback(Py_None);
    return PyModule_Create(&_module);
}