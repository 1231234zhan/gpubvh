
#include "bvh.cuh"
#include <fstream>
#include <iostream>
#include <sstream>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

using namespace std;

const int threadnum = 128;
int blocknum;

void load_obj(Mesh& mesh, const string& filename)
{
    ifstream file(filename);
    if (!file) {
        exit(EXIT_FAILURE);
    }
    while (file) {
        string line, word;
        getline(file, line);
        stringstream linestream(line);
        linestream >> word;
        if (word == "v") {
            double x, y, z;
            linestream >> x >> y >> z;
            Node node(x, y, z);
            mesh.nodes.push_back(node);
        } else if (word == "f") {
            int a[3], foo;
            char bar;
            for (int i = 0; i < 3; i++) {
                linestream >> a[i] >> bar >> foo;
            }
            Face face(a[0] - 1, a[1] - 1, a[2] - 1);
            mesh.faces.push_back(face);
        }
    }
}

void print_ans(Collision* collis)
{
    cout << collis[0].i << endl;
    for (int i = 1; i <= collis[0].i; i++) {
        cout << collis[i].i + 1 << " " << collis[i].j + 1 << endl;
    }
}

Box get_allnode_box(Mesh& mesh)
{
    Box box(mesh.nodes[0]);
    for (int i = 1; i < mesh.nodes.size(); i++) {
        box.update(mesh.nodes[i]);
    }
    return box;
}

void freeBVH(BVH& bvh)
{
    HANDLE_ERROR(cudaFree((void**)&bvh.nodes));
    HANDLE_ERROR(cudaFree((void**)&bvh.faces));
    HANDLE_ERROR(cudaFree((void**)&bvh.mtcode));
    HANDLE_ERROR(cudaFree((void**)&bvh.tnodes));
    HANDLE_ERROR(cudaFree((void**)&bvh.collis));
    delete bvh.h_collis;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        exit(EXIT_FAILURE);
    }
    Mesh mesh;
    load_obj(mesh, string(argv[1]));
    int nnode = mesh.nodes.size();
    int nface = mesh.faces.size();
    if (nnode < 1 || nface < 1) {
        exit(EXIT_FAILURE);
    }
    Box maxbox = get_allnode_box(mesh);
    BVH bvh(mesh.nodes, mesh.faces);

    blocknum = nface / threadnum + 1;

    morton_code<<<blocknum, threadnum>>>(bvh, maxbox);
    HANDLE_ERROR(cudaGetLastError());

    thrust::device_ptr<ulong> d_mt(bvh.mtcode);
    thrust::device_ptr<Face> d_fa(bvh.faces);
    thrust::sort_by_key(d_mt, d_mt + nface, d_fa);

    generate_tree<<<blocknum, threadnum>>>(bvh);
    HANDLE_ERROR(cudaGetLastError());

    get_tree_bbox<<<blocknum, threadnum>>>(bvh);
    HANDLE_ERROR(cudaGetLastError());

    bvh.root = nface;

    // set initial collision num to zero
    bvh.h_collis[0] = Collision(0, 0);
    HANDLE_ERROR(cudaMemcpy(bvh.collis, bvh.h_collis, sizeof(Collision), cudaMemcpyHostToDevice));

    detect_collision<<<blocknum, threadnum>>>(bvh);
    HANDLE_ERROR(cudaGetLastError());

    HANDLE_ERROR(cudaMemcpy(bvh.h_collis, bvh.collis, bvh.nface * sizeof(Collision), cudaMemcpyDeviceToHost));
    print_ans(bvh.h_collis);

    freeBVH(bvh);
    return 0;
}