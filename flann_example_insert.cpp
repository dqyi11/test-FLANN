#include <flann/flann.hpp>
#include <iostream>
#include <vector>

using namespace flann;

int main(int argc, char** argv)
{

    // construct an randomized kd-tree index using 4 kd-trees
 
    float* p1 = new float[2];
    p1[0] = 1.0; p1[1] = 2.0;
    Index<L2<float> > index( Matrix<float>(p1, 1, 2), flann::KDTreeIndexParams(4));
    index.buildIndex();

    std::cout << "add 1" << std::endl;
  
    float* p2 = new float[2];
    p2[0] = 1.5; p2[1] = 2.5;
    index.addPoints( Matrix<float>(p2, 1, 2 ) );
    
    std::cout << "add 2" << std::endl;

    float* p3 = new float[2];
    p3[0] = 3.5; p3[1] = 3.5;
    index.addPoints( Matrix<float>(p3, 1, 2 ) );
     
    std::cout << "add 3" << std::endl;
    
    std::vector<std::vector<int> > indices;
    std::vector<std::vector<float> > dists;
    
    // do a knn search, using 128 checks
    //index.knnSearch(query, indices, dists, nn, flann::SearchParams(128));
    float* query_point= new float[2];
    query_point[0] = 4.0; query_point[1] = 4.0;
    //int ret = index.radiusSearch( Matrix<float>(query_point, 1, 2), indices, dists, 1.0, flann::SearchParams(128));

    int ret = index.knnSearch( Matrix<float>(query_point, 1, 2) , indices, dists, 2, flann::SearchParams(128));
    std::cout << ret << std::endl;
    std::cout << "SIZE " << indices[0].size() << std::endl;
    for(int i=0; i < indices[0].size(); i++ ) {

        std::cout << index.getPoint(indices[0][i])[0] << " " << index.getPoint(indices[0][i])[1] << std::endl;

    }

    return 0;
}
