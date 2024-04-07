//
// Created by david on 05-02-2024.
//

#ifndef RAPTOR_FUNCTION_ARGUMENT_HPP
#define RAPTOR_FUNCTION_ARGUMENT_HPP

namespace raptor {

    struct input {};

    struct output {};

    template <typename Arg>
    struct in : input {};

    template <typename Arg>
    struct out : output {};

    template <typename Arg>
    struct inout : input, output {};
}

#endif //RAPTOR_FUNCTION_ARGUMENT_HPP
