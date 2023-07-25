#include "csrc.h"
#include <Python.h>
#include <opcode.h>
#include <utility>
using namespace frontend_csrc;
StackEffect frontend_csrc::stack_effect(int opcode, int oparg, int jump) {
    switch (opcode) {
    case NOP:
    case EXTENDED_ARG:
        return {0, 0, 0};

    /* Stack manipulation */
    case POP_TOP:
        return {0, 1, 0};
    case ROT_TWO:
        return {0, 2, 2};
    case ROT_THREE:
        return {0, 3, 3};
    case ROT_FOUR:
        return {0, 4, 4};
    case DUP_TOP:
        return {1, 0, 1};
    case DUP_TOP_TWO:
        return {2, 0, 2};

    /* Unary operators */
    case UNARY_POSITIVE:
    case UNARY_NEGATIVE:
    case UNARY_NOT:
    case UNARY_INVERT:
        return {0, 1, 1};

    // heheda: not sure
    case SET_ADD:
    case LIST_APPEND:
        return {-1, 1, 0};
    case MAP_ADD:
        return {-1, 2, 0};

    /* Binary operators */
    case BINARY_POWER:
    case BINARY_MULTIPLY:
    case BINARY_MATRIX_MULTIPLY:
    case BINARY_MODULO:
    case BINARY_ADD:
    case BINARY_SUBTRACT:
    case BINARY_SUBSCR:
    case BINARY_FLOOR_DIVIDE:
    case BINARY_TRUE_DIVIDE:
        return {0, 2, 1};
    case INPLACE_FLOOR_DIVIDE:
    case INPLACE_TRUE_DIVIDE:
        return {0, 2, 1};

    case INPLACE_ADD:
    case INPLACE_SUBTRACT:
    case INPLACE_MULTIPLY:
    case INPLACE_MATRIX_MULTIPLY:
    case INPLACE_MODULO:
        return {0, 2, 1};
    case STORE_SUBSCR:
        return {0, 3, 0};
    case DELETE_SUBSCR:
        return {0, 2, 0};

    case BINARY_LSHIFT:
    case BINARY_RSHIFT:
    case BINARY_AND:
    case BINARY_XOR:
    case BINARY_OR:
        return {0, 2, 1};
    case INPLACE_POWER:
        return {0, 2, 1};
    case GET_ITER:
        return {0, 1, 1};

    case PRINT_EXPR:
        return {0, 1, 0};
    case LOAD_BUILD_CLASS:
        return {0, 0, 1};
    case INPLACE_LSHIFT:
    case INPLACE_RSHIFT:
    case INPLACE_AND:
    case INPLACE_XOR:
    case INPLACE_OR:
        return {0, 2, 1};

    case SETUP_WITH:
        /* 1 in the normal flow.
         * Restore the stack position and push 6 values before jumping to
         * the handler if an exception be raised. */
        return jump ? StackEffect{-1, 0, 6, true, true}
                    : StackEffect{-1, 0, 1, true, true};
    case RETURN_VALUE:
        return {0, 1, 0};
    case IMPORT_STAR:
        return {0, 1, 0, true};
    case SETUP_ANNOTATIONS:
        return {0, 0, 0, true};
    case YIELD_VALUE:
        return {0, 1, 1};
    case YIELD_FROM:
        return {0, 1, 0};
    case POP_BLOCK:
        return {0, 0, 0, false, true};
    case POP_EXCEPT:
        return {0, 3, 0, false, true};

    case STORE_NAME:
        return {0, 1, 0, true};
    case DELETE_NAME:
        return {0, 0, 0, true};
    case UNPACK_SEQUENCE:
        return {0, 1, oparg};
    case UNPACK_EX:
        return {1, 0, (oparg & 0xFF) + (oparg >> 8)}; // heheda: not sure
    case FOR_ITER:
        /* -1 at end of iterator, 1 if continue iterating. */
        return jump > 0 ? StackEffect{0, 1, 0} : StackEffect{0, 1, 2};

    case STORE_ATTR:
        return {0, 2, 0};
    case DELETE_ATTR:
        return {0, 1, 0};
    case STORE_GLOBAL:
        return {0, 1, 0, false, true};
    case DELETE_GLOBAL:
        return {0, 0, 0, false, false};
    case LOAD_CONST:
        return {0, 0, 1};
    case LOAD_NAME:
        return {0, 0, 1, false};
    case BUILD_TUPLE:
    case BUILD_LIST:
    case BUILD_SET:
    case BUILD_STRING:
        return {0, oparg, 1};
    case BUILD_MAP:
        return {0, 2 * oparg, 1};
    case BUILD_CONST_KEY_MAP:
        return {0, oparg + 1, 1};
    case LOAD_ATTR:
        return {0, 1, 1};
    case COMPARE_OP:
    case IS_OP:
    case CONTAINS_OP:
        return {0, 2, 1};
    case JUMP_IF_NOT_EXC_MATCH:
        return {0, 2, 0};
    case IMPORT_NAME:
        return {0, 2, 1};
    case IMPORT_FROM:
        return {0, 0, 1};

    /* Jumps */
    case JUMP_FORWARD:
    case JUMP_ABSOLUTE:
        return {0, 0, 0};

    case JUMP_IF_TRUE_OR_POP:
    case JUMP_IF_FALSE_OR_POP:
        return jump ? StackEffect{1, 0, 0} : StackEffect{0, 1, 0};

    case POP_JUMP_IF_FALSE:
    case POP_JUMP_IF_TRUE:
        return {0, 1, 0};

    case LOAD_GLOBAL:
        return {0, 0, 1, false, true};

    /* Exception handling */
    case SETUP_FINALLY:
        /* 0 in the normal flow.
         * Restore the stack position and push 6 values before jumping to
         * the handler if an exception be raised. */
        return jump ? StackEffect{0, -6, 0} : StackEffect{0, 0, 0};
    case RERAISE:
        return {0, 3, 0};

    case WITH_EXCEPT_START:
        return {0, 7, 4};

    case LOAD_FAST:
        return {0, 0, 1};
    case STORE_FAST:
        return {0, 1, 0, true};
    case DELETE_FAST:
        return {0, 0, 0, true};

    case RAISE_VARARGS:
        if (oparg == 0)
            return {0, 0, 0, false, true};
        if (oparg == 1)
            return {0, 1, 0, false, false};
        if (oparg == 2)
            return {0, 2, 0, false, true};
        return {PY_INVALID_STACK_EFFECT, PY_INVALID_STACK_EFFECT,
                PY_INVALID_STACK_EFFECT};

    /* Functions and calls */
    case CALL_FUNCTION:
        return {0, oparg + 1, 1};
    case CALL_METHOD:
        return {0, oparg + 2, 1};
    case CALL_FUNCTION_KW:
        return {0, oparg + 2, 1};
    case CALL_FUNCTION_EX:
        return {0, 2 + ((oparg & 0x01) != 0), 1};
    case MAKE_FUNCTION:
        return {0,
                2 + ((oparg & 0x01) != 0) + ((oparg & 0x02) != 0) +
                    ((oparg & 0x04) != 0) + ((oparg & 0x08) != 0),
                1};
    case BUILD_SLICE:
        if (oparg == 3)
            return {0, 3, 1};
        else
            return {0, 2, 1};

    /* Closures */
    case LOAD_CLOSURE:
        return {0, 0, 1};
    case LOAD_DEREF:
    case LOAD_CLASSDEREF:
        return {0, 0, 1};
    case STORE_DEREF:
        return {0, 1, 0, false, true};
    case DELETE_DEREF:
        return {0, 0, 0, false, true};

    /* Iterators and generators */
    case GET_AWAITABLE:
        return {0, 1, 1};
    case SETUP_ASYNC_WITH:
        /* 0 in the normal flow.
         * Restore the stack position to the position before the result
         * of __aenter__ and push 6 values before jumping to the handler
         * if an exception be raised. */
        return {0, 0, jump ? -1 + 6 : 0};
    case BEFORE_ASYNC_WITH:
        return {0, 1, 2};
    case GET_AITER:
        return {0, 1, 1};
    case GET_ANEXT:
        return {0, 1, 1};
    case GET_YIELD_FROM_ITER:
        return {0, 1, 1};
    case END_ASYNC_FOR:
        return {-1, 7, 0}; // seems related with tos? (if tos is
                           // StopAsyncIteration, pop 7 values)
    case FORMAT_VALUE:
        /* If there's a fmt_spec on the stack, we go from 2->1,
           else 1->1. */
        return {0, (oparg & FVS_MASK) == FVS_HAVE_SPEC ? 2 : 1, 1};
    case LOAD_METHOD:
        return {0, 1, 2};
    case LOAD_ASSERTION_ERROR:
        return {0, 0, 1};
    case LIST_TO_TUPLE:
        return {0, 1, 1};
    case LIST_EXTEND:
    case SET_UPDATE:
    case DICT_MERGE:
    case DICT_UPDATE:
        return {0, 2, 1};
    default:
        return {PY_INVALID_STACK_EFFECT, PY_INVALID_STACK_EFFECT,
                PY_INVALID_STACK_EFFECT};
    }
    return {PY_INVALID_STACK_EFFECT, PY_INVALID_STACK_EFFECT,
            PY_INVALID_STACK_EFFECT}; /* not reachable */
}