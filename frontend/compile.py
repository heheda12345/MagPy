from frontend.c_api import set_eval_frame, set_skip_files
import dis

def optimize_and_run_frame(frame):
    try:
        print(f"interpret_frame {frame.f_code.co_filename}")
        print("bytecode", list(dis.get_instructions(frame.f_code)))
        return 1
    except Exception as e:
        print(e)


def compile(f):
    if not hasattr(compile, "skip_file_setted"):
        set_skip_files(set())
        compile.skip_file_setted = True
    def _fn(*args, **kwargs):
        prior = set_eval_frame(optimize_and_run_frame)
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print(e)
        finally:
            print("restoring frame")
            print("prior:", prior)
            set_eval_frame(prior)
    return _fn
        