#include "entities.h"

Tetrahedron::Tetrahedron(const CubeType& cube_points, const std::map<CoordType, int>& points, const int tet_number)
{
    // get tetrahedrons
    switch(tet_number){
        case 0:
        {
            cube_id2point_id = {0, 1, 3, 4};
            break;
        }
        case 1:
        {
            cube_id2point_id = {1, 2, 3, 6};
            break;
        }
        case 2:
        {
            cube_id2point_id = {4, 5, 6, 1};
            break;
        }
        case 3:
        {
            cube_id2point_id = {4, 7, 6, 3};
            break;
        }
        case 4:
        {
            cube_id2point_id = {4, 6, 1, 3};
            break;
        }
    }
    std::vector<CoordType> coords;
    for (int i = 0; i < 4; i++) coords.push_back(cube_points[cube_id2point_id[i]]);
    coordinates = coords;

    // get facets for tetrahedron
    std::vector<std::vector<int>> facet_ids;
    switch (tet_number){
        case 0:
        {
            facet_ids = {
                {0, 1, 4},
                {0, 4, 3},
                {1, 0, 3},
                {4, 1, 3}
            };
            break;
        }
        case 1:
        {
            facet_ids = {
                {1, 2, 6},
                {6, 2, 3},
                {1, 2, 3},
                {1, 6, 3}
            };
            break;
        }
        case 2:
        {
            facet_ids = {
                {1, 5, 4},
                {1, 6, 5},
                {4, 5, 6},
                {4, 6, 1}
            };
            break;
        }
        case 3:
        {
            facet_ids = {
                {4, 7, 3},
                {3, 7, 6},
                {4, 7, 6},
                {4, 6, 3}
            };
            break;
        }
        case 4:
        {
            facet_ids = {
                {4, 3, 1},
                {1, 3, 6},
                {4, 1, 6},
                {4, 5, 6}
            };
            break;
        }
    }

    // for (int f_id = 0; f_id < 4; f_id++){
    //     std::vector<int> _local_facet;
    //     for (int idx = 0; idx < 3; idx++) {
    //         auto it = std::find(cube_id2point_id.begin(), cube_id2point_id.end(), facet_ids[f_id][idx]);
    //         int pos = std::distance(cube_id2point_id.begin(), it);
    //         _local_facet.push_back(pos);
    //     }
    //     facets.push_back(_local_facet);
    // }

    // generate ids of boundary facets
    for (int i = 0; i < 4; i++){
        FacetType facet = {coordinates[facets[i][0]], coordinates[facets[i][1]], coordinates[facets[i][2]]};
        int num_boundary = 0;
        for (auto& coord : facet){
            if (Tetrahedron::is_boundary_point(points, coord)) num_boundary++;
    }
    if (num_boundary == 3) boundary_facets.push_back(i);
    }
}

std::vector<CoordType> Tetrahedron::get_points() { return coordinates; }
std::vector<FacetType> Tetrahedron::get_facets() {
    std::vector<FacetType> tfacets = {
        {coordinates[facets[0][0]], coordinates[facets[0][1]], coordinates[facets[0][2]]},
        {coordinates[facets[1][0]], coordinates[facets[1][1]], coordinates[facets[1][2]]},
        {coordinates[facets[2][0]], coordinates[facets[2][1]], coordinates[facets[2][2]]},
        {coordinates[facets[3][0]], coordinates[facets[3][1]], coordinates[facets[3][2]]}};
    return tfacets;
}

std::vector<FacetType> Tetrahedron::get_boundary_facets() {
    std::vector<FacetType> _facets = Tetrahedron::get_facets();
    std::vector<FacetType> _bfacets;
    int n_bfacets = Tetrahedron::boundary_facets.size();
    for (int i = 0; i < n_bfacets; i++) _bfacets.push_back(_facets[i]);
    return _bfacets;
}

bool Tetrahedron::is_boundary_point(const std::map<std::vector<int>, int>& all_points, std::vector<int> check_point){
    int num_neighbors = 0;
    int num_neighbors_diag = 0;
    int i, j, k;
    i = check_point[0];
    j = check_point[1];
    k = check_point[2];

    std::vector<std::vector<int>> neighbor_points = {
        {i + 1, j, k},
        {i, j + 1, k},
        {i, j, k + 1},
        {i - 1, j, k},
        {i, j - 1, k},
        {i, j, k - 1},
    };
    std::vector<std::vector<int>> neighbor_points_diag = {
        {i, j + 1, k + 1},
        {i, j - 1, k - 1},
        {i, j - 1, k + 1},
        {i, j + 1, k - 1},

        {i + 1, j, k + 1},
        {i - 1, j, k - 1},
        {i - 1, j, k + 1},
        {i + 1, j, k - 1},

        {i + 1, j + 1, k},
        {i - 1, j - 1, k},
        {i - 1, j + 1, k},
        {i + 1, j - 1, k},
    };

    for (int idx = 0; idx < 6; idx++){
        if (all_points.count(neighbor_points[idx]) > 0){
            num_neighbors++;
        }
    }
    for (int idx = 0; idx < 12; idx++){
        if (all_points.count(neighbor_points_diag[idx]) > 0){
            num_neighbors_diag++;
        }
    }
    return num_neighbors != 6 || num_neighbors_diag != 12;
}

// std::vector<std::vector<int>> Tetrahedron::get_boundary_facets_ids(std::vector<int> point_ids){}
//     std::vector<std::vector<int>> _local_bfacets;
//     std::vector<std::vector<int>> boundary_facet_ids;
//     for (auto& vec : Tetrahedron::boundary_facets){
//         _local_bfacets.push_back(Tetrahedron::facets[vec]);
//     }

//     for (auto& vec : _local_bfacets){
//         std::vector<int> _lf_ids;
//         for (auto& elem : vec) _lf_ids.push_back(point_ids[])
//     }

// }