/* meshconv IN OUT FORMAT_ID -- convert a mesh with assimp (e.g. FORMAT_ID = fbx).
 * Built by tools/build_meshconv.sh against the assimp library in .tools/assimp. */
#include <assimp/cexport.h>
#include <assimp/cimport.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s IN OUT FORMAT_ID\n", argv[0]); return 2; }
    const struct aiScene *scene = aiImportFile(argv[1], aiProcess_JoinIdenticalVertices);
    if (!scene) { fprintf(stderr, "import failed: %s\n", aiGetErrorString()); return 1; }
    if (aiExportScene(scene, argv[3], argv[2], 0) != AI_SUCCESS) {
        fprintf(stderr, "export failed: %s\n", aiGetErrorString());
        aiReleaseImport(scene);
        return 1;
    }
    unsigned verts = 0;
    for (unsigned m = 0; m < scene->mNumMeshes; m++) verts += scene->mMeshes[m]->mNumVertices;
    printf("%s -> %s (%s): %u meshes, %u vertices\n", argv[1], argv[2], argv[3], scene->mNumMeshes, verts);
    aiReleaseImport(scene);
    return 0;
}
