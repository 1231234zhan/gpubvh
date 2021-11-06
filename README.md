
Bounding volume hierarchy implemented in CUDA, used to detect face collisions.

The algorithm's idea mainly comes from [https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/)


To build and run:

```
make
./bvh {obj-file}
```

Note that in the obj file, faces should present in such format 
```
f {a}/{b} {c}/{d} {e}/{f}
```