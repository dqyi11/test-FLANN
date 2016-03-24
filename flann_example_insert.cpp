#include <flann/flann.hpp>
#include <iostream>
#include <vector>

using namespace flann;

int main(int argc, char** argv)
{

    // construct an randomized kd-tree index using 4 kd-trees
    Index<L2<float> > index(flann::KDTreeIndexParams(4));
    float* point1 = new float[2];
    point1[0] = 1.0; point1[1] = 2.0;
    index.addPoints( Matrix<float>(point1, 1, 2) );

    float* point2 = new float[2];
    point2[0] = 1.5; point2[1] = 2.5;
    index.addPoints( Matrix<float>(point2, 1, 2) );
     
    std::vector<std::vector<int> > indices;
    std::vector<std::vector<float> > dists;
    
    // do a knn search, using 128 checks
    //index.knnSearch(query, indices, dists, nn, flann::SearchParams(128));
    float query_point[2];
    query_point[0] = 1.0; query_point[1] = 2.0;
    int ret = index.radiusSearch( Matrix<float>(query_point, 1, 2), indices, dists, 2.0, flann::SearchParams(128));

    std::cout << ret << std::endl;
    std::cout << "SIZE " << indices[0].size() << std::endl;
    for(int i=0; i < indices[0].size(); i++ ) {

        std::cout << index.getPoint(indices[0][i])[0] << " " << index.getPoint(indices[0][i])[1] << std::endl;

    }

    return 0;
}
