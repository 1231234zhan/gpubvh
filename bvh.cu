
#include "bvh.cuh"
#include "tri_contact.cuh"

__host__ __device__ Box::Box(const BVH& bvh, const Face& face)
{
    Box box(bvh.nodes[face.nid[0]]);
    box.update(bvh.nodes[face.nid[1]]);
    box.update(bvh.nodes[face.nid[2]]);
    *this = box;
}

void Box::update(const Box& box_j)
{
    x[0] = min(x[0], box_j.x[0]);
    y[0] = min(y[0], box_j.y[0]);
    z[0] = min(z[0], box_j.z[0]);
    x[1] = max(x[1], box_j.x[1]);
    y[1] = max(y[1], box_j.y[1]);
    z[1] = max(z[1], box_j.z[1]);
}

void Box::update(const Node& node_j)
{
    x[0] = min(x[0], node_j.x);
    y[0] = min(y[0], node_j.y);
    z[0] = min(z[0], node_j.z);
    x[1] = max(x[1], node_j.x);
    y[1] = max(y[1], node_j.y);
    z[1] = max(z[1], node_j.z);
}

const int bitlimit = 40;
__device__ ulong morton3D(const Node& node, const Box& mbox)
{

    Node boxnode(mbox.x[1] - mbox.x[0], mbox.y[1] - mbox.y[0], mbox.z[1] - mbox.z[0]);
    Node newnode(node.x - mbox.x[0], node.y - mbox.y[0], node.z - mbox.z[0]);
    ulong mtcd = 0;
    for (int i = 0; i < bitlimit; i++) {
        int ax = 0;
        for (int j = 1; j < 3; j++) {
            if (boxnode.c[j] > boxnode.c[ax])
                ax = j;
        }
        boxnode.c[ax] /= 2;
        mtcd = mtcd << 1;
        if (newnode.c[ax] > boxnode.c[ax]) {
            newnode.c[ax] -= boxnode.c[ax];
            mtcd = mtcd | 1;
        }
    }
    return mtcd;
}

__device__ Node get_barycentric_coords(const BVH& bvh, const Face& face)
{
    Node node(0.0, 0.0, 0.0);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            const Node& foo_node = bvh.nodes[face.nid[j]];
            node.c[i] += foo_node.c[i];
        }
        node.c[i] /= 3;
    }
    return node;
}

// Morton code
// https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/

__global__ void morton_code(BVH bvh, Box mbox)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= bvh.nface)
        return;

    Node cnode = get_barycentric_coords(bvh, bvh.faces[tid]);
    bvh.mtcode[tid] = morton3D(cnode, mbox);
}

__device__ inline int pfx(ulong a, ulong b)
{
    return __clzll(a ^ b);
}

__device__ inline int sign(int a)
{
    return a > 0 ? 1 : (a < 0 ? -1 : 0);
}

__device__ inline bool inside(int a, int b, int c)
{
    return a >= b && a <= c;
}

// Magical parallel tree bulding
// https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2012/11/karras2012hpg_paper.pdf

__global__ void generate_tree(BVH bvh)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= bvh.nface - 1)
        return;

    auto mtcode = bvh.mtcode;
    int d = tid == 0 ? 1 : sign(pfx(mtcode[tid], mtcode[tid + 1]) - pfx(mtcode[tid], mtcode[tid - 1]));
    int minpfx = tid == 0 ? -1 : pfx(mtcode[tid], mtcode[tid - d]);

    int lmax = 2, ci, cj;
    ci = cj = tid;
    while (inside(ci + d * lmax, 0, bvh.nface - 1) && pfx(mtcode[ci], mtcode[ci + d * lmax]) > minpfx)
        lmax *= 2;

    lmax /= 2;
    while (lmax > 0) {
        if (inside(cj + d * lmax, 0, bvh.nface - 1) && pfx(mtcode[ci], mtcode[cj + d * lmax]) > minpfx)
            cj += d * lmax;
        lmax /= 2;
    }

    minpfx = pfx(mtcode[ci], mtcode[cj]);
    int ck, L, R, mid;
    L = ci, R = cj + d;
    while (1 < d * (R - L)) {
        mid = (L + R) / 2;
        if (pfx(mtcode[ci], mtcode[mid]) > minpfx)
            L = mid;
        else
            R = mid;
    }
    ck = L;

    int offset = bvh.nface;
    ck += min(d, 0);
    auto tnodes = bvh.tnodes;
    tnodes[offset + tid].isleaf = false;
    tnodes[offset + tid].ci = min(ci, cj);
    tnodes[offset + tid].cj = max(ci, cj);

    if (min(ci, cj) == ck) {
        BVHnode tnode(true, ck, -1, -1, offset + tid);
        tnode.bbox = Box(bvh, bvh.faces[ck]);
        tnodes[ck] = tnode;
        tnodes[offset + tid].lchild = ck;
    } else {
        tnodes[offset + tid].lchild = offset + ck;
        tnodes[offset + ck].parent = offset + tid;
    }

    if (max(ci, cj) == ck + 1) {
        BVHnode tnode(true, ck + 1, -1, -1, offset + tid);
        tnode.bbox = Box(bvh, bvh.faces[ck + 1]);
        tnodes[ck + 1] = tnode;
        tnodes[offset + tid].rchild = ck + 1;
    } else {
        tnodes[offset + tid].rchild = offset + ck + 1;
        tnodes[offset + ck + 1].parent = offset + tid;
    }

    if (tid == 0)
        tnodes[offset + tid].parent = -1;

    // bounding box calculation
    // leafid = 0 means that node has not been visited yet
    tnodes[offset + tid].leafid = 0;
}

__global__ void get_tree_bbox(BVH bvh)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= bvh.nface)
        return;

    // begin with leaf node
    BVHnode* tnodes = bvh.tnodes;
    int pa = tnodes[tid].parent;
    Box box = tnodes[tid].bbox;
    while (pa >= 0) {
        int flag = atomicExch(&(tnodes[pa].leafid), 1);
        if (flag == 0)
            break;
        int lc = tnodes[pa].lchild;
        int rc = tnodes[pa].rchild;

        box.update(tnodes[lc].bbox);
        box.update(tnodes[rc].bbox);
        tnodes[pa].bbox = box;
        pa = tnodes[pa].parent;
    }
}

__device__ bool detect_face_colli(int a, int b, const BVH& bvh)
{
    Face& face_a = bvh.faces[a];
    Face& face_b = bvh.faces[b];

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (face_a.nid[i] == face_b.nid[j])
                return false;

    vec3f v1(bvh.nodes[face_a.nid[0]].c);
    vec3f v2(bvh.nodes[face_a.nid[1]].c);
    vec3f v3(bvh.nodes[face_a.nid[2]].c);
    vec3f v4(bvh.nodes[face_b.nid[0]].c);
    vec3f v5(bvh.nodes[face_b.nid[1]].c);
    vec3f v6(bvh.nodes[face_b.nid[2]].c);

    return tri_contact(v1, v2, v3, v4, v5, v6);
}

__device__ bool detect_box_colli(const Box& box_i, const Box& box_j)
{
    if (box_i.x[1] < box_j.x[0] || box_i.y[1] < box_j.y[0] || box_i.z[1] < box_j.z[0])
        return false;
    if (box_i.x[0] > box_j.x[1] || box_i.y[0] > box_j.x[1] || box_i.z[0] > box_j.z[1])
        return false;
    return true;
}

const int stacksize = 64;
__global__ void detect_collision(BVH bvh)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= bvh.nface)
        return;

    Face face_i = bvh.faces[tid];
    auto tnodes = bvh.tnodes;
    Box box_i(bvh, face_i);
    int stack[stacksize];
    int ptr = 0;

    stack[ptr++] = bvh.root;

    while (ptr != 0) {
        int id = stack[--ptr];

        if (tnodes[id].isleaf) {
            if (detect_face_colli(tid, tnodes[id].leafid, bvh)) {
                int colid = atomicAdd(&(bvh.collis[0].i), 1);
                if (colid + 5 < bvh.nface)
                    bvh.collis[colid + 1] = Collision(tid, tnodes[id].leafid);
            }
        } else {
            int lc = tnodes[id].lchild;
            int rc = tnodes[id].rchild;
            if (tid <= tnodes[lc].cj && detect_box_colli(box_i, tnodes[lc].bbox)) {
                stack[ptr++] = lc;
            }

            if (tid <= tnodes[rc].cj && detect_box_colli(box_i, tnodes[rc].bbox)) {
                stack[ptr++] = rc;
            }
        }
    }
}