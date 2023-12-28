#include "csrc.h"
#include <Python.h>
#include <descrobject.h>
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

typedef struct {
    PyObject_HEAD PyObject *mapping;
} mappingproxyobject;

PyObject *parse_mapproxyobject(PyObject *self, PyObject *args) {
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }
    if (Py_TYPE(obj) != &PyDictProxy_Type) {
        PyErr_SetString(PyExc_TypeError, "Expected mapproxyobject");
        return NULL;
    }
    mappingproxyobject *mobj = (mappingproxyobject *)obj;
    Py_INCREF(mobj->mapping);
    return mobj->mapping;
}

typedef struct {
    PyObject_HEAD PyObject *iters;
    PyObject *func;
} mapobject;

PyObject *parse_mapobject(PyObject *self, PyObject *args) {
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }
    if (Py_TYPE(obj) != &PyMap_Type) {
        PyErr_SetString(PyExc_TypeError, "Expected mapobject");
        return NULL;
    }
    mapobject *mobj = (mapobject *)obj;
    Py_INCREF(mobj->iters);
    Py_INCREF(mobj->func);
    return PyTuple_Pack(2, mobj->iters, mobj->func);
}

PyObject *parse_cell(PyObject *self, PyObject *args) {
    PyObject *cell;
    if (!PyArg_ParseTuple(args, "O", &cell)) {
        return NULL;
    }
    if (Py_TYPE(cell) != &PyCell_Type) {
        PyErr_SetString(PyExc_TypeError, "Expected cell");
        return NULL;
    }
    PyCellObject *cobj = (PyCellObject *)cell;
    if (cobj->ob_ref == NULL) {
        PyObject *null_obj = NullObjectSingleton::getInstance().getNullObject();
        Py_INCREF(null_obj);
        return null_obj;
    }
    Py_INCREF(cobj->ob_ref);
    return cobj->ob_ref;
}

PyObject *set_cell(PyObject *self, PyObject *args) {
    PyObject *cell, *value;
    if (!PyArg_ParseTuple(args, "OO", &cell, &value)) {
        return NULL;
    }
    if (Py_TYPE(cell) != &PyCell_Type) {
        PyErr_SetString(PyExc_TypeError, "Expected cell");
        return NULL;
    }
    PyCell_SET(cell, value);
    return Py_None;
}

} // namespace frontend_csrc