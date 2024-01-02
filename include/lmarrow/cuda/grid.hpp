//
// Created by david on 31-10-2023.
//

#ifndef GDGRAPH_GRID_HPP
#define GDGRAPH_GRID_HPP

//#define def_tpb(size) 1024
//#define def_tpb(size) \
//    ((size < 512) ? 32 : \
//    (size < 2048) ? 64 : \
//    (size < 4096) ? 128 : \
//    (size < 8192) ? 512 : \
//    1024)

#define def_tpb(size) \
    ((size < 4096) ? 128 : \
    (size < 8192) ? 256 : \
    (size < 16384) ? 512 : \
    1024)

#define def_nb(size) ( ((size) + def_tpb(size) - 1) / def_tpb(size) )

#endif //GDGRAPH_GRID_HPP
