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
    ~Tetrahedron(){
        coordinates.clear(); coordinates.shrink_to_fit();
        facets.clear(); facets.shrink_to_fit();
        boundary_facets.clear(); boundary_facets.shrink_to_fit();
        cube_id2point_id.clear(); cube_id2point_id.shrink_to_fit();
    };
    std::vector<FacetType> get_facets();
    std::vector<FacetType> get_boundary_facets();
    std::vector<CoordType> get_points();
    std::vector<std::vector<int>> get_boundary_facets_ids(std::vector<int>);
};

class Cube {
    std::vector<CoordType> points;
public:
    Cube(const std::vector<CoordType>&, std::map<CoordType, int>&);
    ~Cube(){
        points.clear(); points.shrink_to_fit();
    };
    std::vector<std::vector<CoordType&>> generate_external_triangles(bool use_half_cubes);
    std::vector<std::vector<CoordType&>> generate_tetrahedrons();
    void print_mesh_statistics();
};