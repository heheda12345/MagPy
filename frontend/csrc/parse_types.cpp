#include <Python.h>
#include <object.h>

namespace frontend_csrc {

typedef struct {
    PyObject_HEAD long index;
    long start;
    long step;
    long len;
} rangeiterobject;

PyObject *parse_rangeiterobject(PyObject *self, PyObject *args) {
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }
    if (Py_TYPE(obj) != &PyRangeIter_Type) {
        PyErr_SetString(PyExc_TypeError, "Expected rangeiterobject");
        return NULL;
    }
    rangeiterobject *robj = (rangeiterobject *)obj;
    return PyTuple_Pack(
        4, PyLong_FromLong(robj->index), PyLong_FromLong(robj->start),
        PyLong_FromLong(robj->step), PyLong_FromLong(robj->len));
}

PyObject *make_rangeiterobject(PyObject *self, PyObject *args) {
    long index, start, step, len;
    if (!PyArg_ParseTuple(args, "llll", &index, &start, &step, &len)) {
        return NULL;
    }
    rangeiterobject *robj = PyObject_New(rangeiterobject, &PyRangeIter_Type);
    robj->index = index;
    robj->start = start;
    robj->step = step;
    robj->len = len;
    return (PyObject *)robj;
}

} // namespace frontend_csrc