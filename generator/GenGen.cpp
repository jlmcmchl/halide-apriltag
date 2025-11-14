#include "Halide.h"
#include <exception>
#include <cstdlib>

namespace {
    // Terminate handler to catch exceptions that bypass normal catch blocks
    // (e.g., exceptions thrown from destructors during stack unwinding)
    // Note: When std::terminate is called, the exception may not always be
    // available via std::current_exception(), but we try to catch it anyway.
    void terminate_handler() {
        std::exception_ptr ex = std::current_exception();
        if (ex) {
            try {
                std::rethrow_exception(ex);
            } catch (const Halide::CompileError &e) {
                std::cerr << "Halide CompileError caught in terminate handler: " << e.what() << "\n";
                std::abort();
            } catch (const Halide::Error &err) {
                std::cerr << "Halide Error caught in terminate handler: " << err.what() << "\n";
                std::abort();
            } catch (const std::exception &err) {
                std::cerr << "Exception caught in terminate handler: " << err.what() << "\n";
                std::abort();
            } catch (...) {
                std::cerr << "Unknown exception caught in terminate handler\n";
                std::abort();
            }
        } else {
            // std::terminate was called but no exception is available
            // This can happen when an exception is thrown from a destructor
            // during stack unwinding, or from noexcept code
            std::cerr << "std::terminate called (exception may have been thrown from destructor or noexcept code)\n";
            std::abort();
        }
    }
}

int main(int argc, char **argv) {
    // Set up terminate handler to catch exceptions that bypass normal catch blocks
    std::set_terminate(terminate_handler);
    
    try {
        return Halide::Internal::generate_filter_main(argc, argv);
    } catch (const Halide::CompileError &e) {
        fprintf(stderr, "Halide GPU JIT compile error: %s\n", e.what());
        return -1;
    } catch (const Halide::Error &err) {
        // Do *not* use user_error here (or elsewhere in this function): it
        // will throw an exception, and since there is almost certainly no
        // try/catch block in our caller, it will call std::terminate,
        // swallowing all error messages.
        std::cerr << "Unhandled exception: " << err.what() << "\n";
        return -1;
    } catch (const std::exception &err) {
        std::cerr << "Unhandled exception: " << err.what() << "\n";
        return -1;
    } catch (...) {
        std::cerr << "Unhandled exception: (unknown)\n";
        return -1;
    }
}