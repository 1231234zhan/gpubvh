
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
            face.id = mesh.faces.size();
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
    HANDLE_ERROR(cudaFree(bvh.nodes));
    HANDLE_ERROR(cudaFree(bvh.faces));
    HANDLE_ERROR(cudaFree(bvh.mtcode));
    HANDLE_ERROR(cudaFree(bvh.tnodes));
    HANDLE_ERROR(cudaFree(bvh.collis));
    delete bvh.h_collis;
}

struct Timer {
    cudaEvent_t start, stop;
    Timer()
    {
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
    }
    ~Timer()
    {
        HANDLE_ERROR(cudaEventDestroy(start));
        HANDLE_ERROR(cudaEventDestroy(stop));
    }
    void begin()
    {
        HANDLE_ERROR(cudaEventRecord(start, 0));
    }
    void print()
    {
        HANDLE_ERROR(cudaEventRecord(stop, 0));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        float elapsedTime;
        HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
            start, stop));
        printf("Time to generate:  %3.1f ms\n", elapsedTime);
    }
};

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

    // Start timer before malloc and memcpy
    Timer timer;
    timer.begin();
    BVH bvh(mesh.nodes, mesh.faces);

    blocknum = nface / threadnum + 1;

    morton_code<<<blocknum, threadnum>>>(bvh, maxbox);
    HANDLE_ERROR(cudaGetLastError());

    timer.print();

    thrust::device_ptr<ulong> d_mt(bvh.mtcode);
    thrust::device_ptr<Face> d_fa(bvh.faces);
    thrust::sort_by_key(d_mt, d_mt + nface, d_fa);

    timer.print();

    generate_tree<<<blocknum, threadnum>>>(bvh);
    HANDLE_ERROR(cudaGetLastError());

    timer.print();

    get_tree_bbox<<<blocknum, threadnum>>>(bvh);
    HANDLE_ERROR(cudaGetLastError());

    timer.print();

    bvh.root = nface;

    // set initial collision num to zero
    bvh.h_collis[0] = Collision(0, 0);
    HANDLE_ERROR(cudaMemcpy(bvh.collis, bvh.h_collis, sizeof(Collision), cudaMemcpyHostToDevice));

    detect_collision<<<blocknum, threadnum>>>(bvh);
    HANDLE_ERROR(cudaGetLastError());

    HANDLE_ERROR(cudaMemcpy(bvh.h_collis, bvh.collis, bvh.nface * sizeof(Collision), cudaMemcpyDeviceToHost));
    
    timer.print();

    print_ans(bvh.h_collis);

    freeBVH(bvh);
    return 0;
}