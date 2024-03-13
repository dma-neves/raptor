//
// Created by david on 12-03-2024.
//

#ifndef GDGRAPH_ERROR_CHECK_HPP
#define GDGRAPH_ERROR_CHECK_HPP

#define cudaErrorCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort =
true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "cudaAssert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

#endif //GDGRAPH_ERROR_CHECK_HPP
