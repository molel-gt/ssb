#include <map>
#include <ranges>
#include <vector>

typedef std::vector<int> CoordType;
typedef std::vector<std::vector<int>> CubeType;
typedef std::vector<std::vector<int>> FacetType;
typedef std::vector<std::vector<int>> TetrahedronType;

class Tetrahedron {
    /*
    Accepts: 
        `cube_points`: points ordered as 000,100,110,010,001,101,111,011 using the representative unit cube.
        `ten_number`: which tetrahedron to extract from the cube
        `points`: mapping of coordinates of entire domain to integer
    */
    std::vector<CoordType> coordinates;
    std::vector<std::vector<int>> facets;
    std::vector<int> boundary_facets;
    std::vector<int> cube_id2point_id;
    bool is_boundary_point(const std::map<std::vector<int>, int>&, std::vector<int>);
public:
    Tetrahedron(const CubeType&, const std::map<CoordType, int>&, const int);
    std::vector<FacetType> get_facets();
    std::vector<FacetType> get_boundary_facets();
    std::vector<CoordType> get_points();
};