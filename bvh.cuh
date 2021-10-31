#ifndef BVH_CUH
#define BVH_CUH
#include <iostream>
#include <vector>
using namespace std;

#include <cstdio>
static void HandleError(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

struct BVH;

struct Node {
    union {
        struct {
            double c[3];
        };
        struct {
            double x, y, z;
        };
    };

    __host__ __device__ Node() { }
    __host__ __device__ Node(double x, double y, double z)
    {
        c[0] = x;
        c[1] = y;
        c[2] = z;
    }
};

struct Face {
    int nid[3];
    __host__ __device__ Face() { }
    __host__ __device__ Face(int a, int b, int c)
    {
        nid[0] = a;
        nid[1] = b;
        nid[2] = c;
    }
};

struct Mesh {
    std::vector<Face> faces;
    std::vector<Node> nodes;
};

struct Box {
    double x[2], y[2], z[2];
    __host__ __device__ Box() { }
    __host__ __device__ Box(const Node& node)
    {
        x[0] = x[1] = node.c[0];
        y[0] = y[1] = node.c[1];
        z[0] = z[1] = node.c[2];
    }
    __host__ __device__ Box(const BVH&, const Face&);

    __host__ __device__ void update(const Node&);
    __host__ __device__ void update(const Box&);

    // debug
    friend ostream& operator<<(ostream& os, const Box& box)
    {
        os << box.x[0] << " " << box.x[1] << " " << box.y[0] << " " << box.y[1] << " " << box.z[0] << " " << box.z[1];
        return os;
    }
};

struct BVHnode {
    bool isleaf;
    int leafid; // leaf index, faces
    int ci, cj; // left and right range, faces
    int lchild, rchild, parent; //  -1 represent NULL, tnodes
    Box bbox;
    __host__ __device__ BVHnode() { }
    __host__ __device__ BVHnode(bool isleaf, int leafid, int lchild, int rchild, int parent)
        : isleaf(isleaf)
        , leafid(leafid)
        , lchild(lchild)
        , rchild(rchild)
        , parent(parent)
        , ci(leafid)
        , cj(leafid)
    {
    }

    // debug
    friend ostream& operator<<(ostream& os, const BVHnode& node)
    {
        os << node.isleaf << " " << node.leafid << " " << node.ci << " " << node.cj << " " << node.parent << "    ";
        os << node.bbox;

        return os;
    }
};

struct Collision {
    int i, j;
    __host__ __device__ Collision() { }
    __host__ __device__ Collision(int i, int j)
        : i(i)
        , j(j)
    {
    }
};

struct BVH {
    int nnode, nface;
    int root;

    Collision* h_collis;

    Collision* collis; // collis[0].i : collisions num
    Node* nodes;
    Face* faces;
    ulong* mtcode;
    BVHnode* tnodes;
    BVH() { }
    BVH(const std::vector<Node>& h_nodes, const std::vector<Face>& h_faces)
    {
        nnode = h_nodes.size();
        nface = h_faces.size();
        HANDLE_ERROR(cudaMalloc((void**)&nodes, nnode * sizeof(Node)));
        HANDLE_ERROR(cudaMalloc((void**)&faces, nface * sizeof(Face)));
        HANDLE_ERROR(cudaMalloc((void**)&mtcode, nface * sizeof(ulong)));
        HANDLE_ERROR(cudaMalloc((void**)&tnodes, (nface * 2) * sizeof(BVHnode)));
        HANDLE_ERROR(cudaMalloc((void**)&collis, nface * sizeof(Collision)));

        HANDLE_ERROR(cudaMemcpy(nodes, h_nodes.data(), nnode * sizeof(Node), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(faces, h_faces.data(), nface * sizeof(Face), cudaMemcpyHostToDevice));

        h_collis = new Collision[nface];
    }
};

__global__ void morton_code(BVH, Box);
__global__ void generate_tree(BVH);
__global__ void get_tree_bbox(BVH);
__global__ void detect_collision(BVH);
#endif
