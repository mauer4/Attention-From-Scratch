#include <cstdlib>

#include "attention/core.hpp"

int main() {
    return attention::placeholder() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
