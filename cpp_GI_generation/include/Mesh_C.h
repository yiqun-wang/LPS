/************************************************************
* This file is MODIFIED from a part of the 3D descriptor 
* learning framework by Hanyu Wang(王涵玉)
* https://github.com/jianweiguo/local3Ddescriptorlearning
* The library computes Discrete Geodesic Polar Coordinates
* on a polygonal mesh.
* DGPC file's authors: Eivind Lyche Melvær and Martin Reimers
*
* Author: Yiqun Wang(王逸群)
* https://github.com/yiqun-wang/LPS
************************************************************/

#ifndef DGPC_MESH_H
#define DGPC_MESH_H

#include <limits>
#include <fstream>
#include <cassert>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include "Vector3.h"
#include "libcompcur.h"
#include "mclmcr.h"
#include "matrix.h"
#include "mclcppclass.h"


using namespace std;

namespace GIGen
{

    // Redefine default Point class in OpenMesh
    template<class P>
    struct OpenMeshTraits : OpenMesh::DefaultTraits
    {
        typedef P Point;
		VertexAttributes(OpenMesh::Attributes::Status);	
		FaceAttributes(OpenMesh::Attributes::Status);
		EdgeAttributes(OpenMesh::Attributes::Status);
    };

    using Point = GIGen::Vector3<double>;

    struct VPropHandles
    {
        OpenMesh::VPropHandleT<double> c_max, c_min;
        OpenMesh::VPropHandleT<Point> u_max, u_min, normal, resp, point;
        OpenMesh::VPropHandleT<vector<double>> hks, le, cf;
    };

    // Main mesh class
    class MeshOM : public OpenMesh::TriMesh_ArrayKernelT<OpenMeshTraits<Point>>
    {
    private:

        bool is_vertex_info_computed_ = false;
        int hks_len, le_len;
        string filename;
        VPropHandles vphs;

        bool to_VF_mwArray(mwArray &vertices, mwArray &faces)
        {
            // Dimension check.
            mwArray &v_dims = vertices.GetDimensions();
            mwArray &dim_v_dims = v_dims.GetDimensions();

            mwArray &f_dims = faces.GetDimensions();
            mwArray &dim_f_dims = f_dims.GetDimensions();

            int &&dim_v_dims_1_2 = dim_v_dims(1, 2);
            int &&v_dims_1_1 = v_dims(1, 1);
            int &&v_dims_1_2 = v_dims(1, 2);

            int &&dim_f_dims_1_2 = dim_f_dims(1, 2);
            int &&f_dims_1_1 = f_dims(1, 1);
            int &&f_dims_1_2 = f_dims(1, 2);

            if (!(dim_v_dims_1_2 == 2 && v_dims_1_1 == 3 && v_dims_1_2 == this->n_vertices()) ||
                !(dim_f_dims_1_2 == 2 && f_dims_1_1 == 3 && f_dims_1_2 == this->n_faces()))
            {
                cerr << "Filed to call function to_VF_mwArray(*), dimensions do mot match." << endl;
                exit(0);
            }


            //for (auto &vertex : MeshBase::vertices())
            for (auto &vertex : this->vertices())
            {
                auto &v = this->point(vertex);
                for (size_t i = 0; i < 3; i++)
                {
                    vertices(i + 1, vertex.idx() + 1) = v[i];
                }
            }

            for (auto &face : this->faces())
            {
                size_t i = 0;

                // fv_iter belongs to MeshBase::FaceVertexIter
                for (auto fv_iter = this->fv_begin(face); fv_iter.is_valid(); fv_iter++)
                {
                    faces(i + 1, face.idx() + 1) = fv_iter->idx() + 1;
                    i += 1;
                }
            }

            return true;

        }


    public:
		double diameter;

        // For compatibility to Generator.h
        typedef Point point_type;

        MeshOM(const std::string& filename, const int &hks_len = 16, const int &le_len = 5)
        {
            this->hks_len = hks_len;
			this->le_len = le_len;
            this->read_mesh(filename);
        }

        MeshOM() {}

        bool read_mesh(const std::string& filename)
        {
            bool is_succeeded = OpenMesh::IO::read_mesh(*this, filename);

            if (is_succeeded)
            {
                this->filename = filename;
                this->compute_vertex_info();
            }
            else
                return false;

            return is_succeeded;
        }

        void applied_rotation(Rot_mat<double> rotation_mat)
        {
            for (auto &vertex : this->vertices())
            {
                auto &v = this->point(vertex);
                this->set_point(vertex, rotation_mat * v);
            }
        }

        bool compute_vertex_info()
        {

			mwArray vertices(3, this->n_vertices(), mxDOUBLE_CLASS);
			mwArray faces(3, this->n_faces(), mxDOUBLE_CLASS);

			if (!this->to_VF_mwArray(vertices, faces))
			{
				cerr << "to_VF_mwArray failed!" << endl;
				return false;
			}

			mwArray u_max(3, this->n_vertices(), mxDOUBLE_CLASS);
			mwArray u_min(3, this->n_vertices(), mxDOUBLE_CLASS);
			mwArray curvature_max(1, this->n_vertices(), mxDOUBLE_CLASS);
			mwArray curvature_min(1, this->n_vertices(), mxDOUBLE_CLASS);
			mwArray normal(3, this->n_vertices(), mxDOUBLE_CLASS);
			mwArray hks(this->hks_len, this->n_vertices(), mxDOUBLE_CLASS);
			mwArray le(this->le_len, this->n_vertices(), mxDOUBLE_CLASS);
			mwArray cf(1, this->le_len * 3 + 1, mxDOUBLE_CLASS);
			mwArray filenameArray(this->filename.c_str());
			mwArray desc_len(1, 1, mxDOUBLE_CLASS);
			mwArray le_len(1, 1, mxDOUBLE_CLASS);
			mwArray diameter(1, 1, mxDOUBLE_CLASS);
			mwArray resp(3, this->n_vertices(), mxDOUBLE_CLASS);
			desc_len(1, 1) = this->hks_len;
			le_len(1, 1) = (this->le_len * 3.0 + 1) / 3.0;
			mwArray cf_flag(1, 1, mxLOGICAL_CLASS);
			cf_flag(1, 1) = false;

			compute_curvature(10, u_max, u_min, curvature_max, curvature_min, normal, hks, diameter, resp, le, cf, vertices, faces, filenameArray, desc_len, le_len, cf_flag); //number output

			this->diameter = diameter(1, 1);
			this->add_property(vphs.u_max);
			this->add_property(vphs.u_min);
			this->add_property(vphs.c_max);
			this->add_property(vphs.c_min);
			this->add_property(vphs.normal);
			this->add_property(vphs.hks);
			this->add_property(vphs.resp);
			this->add_property(vphs.le);
			this->add_property(vphs.cf);
			this->add_property(vphs.point);

			for (auto &vertex : this->vertices())
			{
				this->property(vphs.c_max, vertex) = curvature_max(1, vertex.idx() + 1);
				this->property(vphs.c_min, vertex) = curvature_min(1, vertex.idx() + 1);
				this->property(vphs.hks, vertex).resize(this->hks_len);
				this->property(vphs.le, vertex).resize(this->le_len); //need ini..
				for (size_t i = 0; i < 3; i++)
				{
					this->property(vphs.u_max, vertex)[i] = u_max(i + 1, vertex.idx() + 1);
					this->property(vphs.u_min, vertex)[i] = u_min(i + 1, vertex.idx() + 1);
					this->property(vphs.normal, vertex)[i] = normal(i + 1, vertex.idx() + 1);
					this->property(vphs.resp, vertex)[i] = resp(i + 1, vertex.idx() + 1);
					this->property(vphs.point, vertex)[i] = vertices(i + 1, vertex.idx() + 1);
				}

				for (size_t i = 0; i < this->hks_len; i++)
				{
					this->property(vphs.hks, vertex)[i] = hks(i + 1, vertex.idx() + 1);
				}

				for (size_t i = 0; i < this->le_len; i++)
				{
					this->property(vphs.le, vertex)[i] = le(i + 1, vertex.idx() + 1);
				}
			}

			this->is_vertex_info_computed_ = true;
			return true;
            
        }

		mwArray compute_vertex_cf(int mul)
		{

			mwArray vertices(3, this->n_vertices(), mxDOUBLE_CLASS);
			mwArray faces(3, this->n_faces(), mxDOUBLE_CLASS);

			if (!this->to_VF_mwArray(vertices, faces))
			{
				cerr << "to_VF_mwArray failed!" << endl;
				return vertices;
			}

			mwArray u_max(3, this->n_vertices(), mxDOUBLE_CLASS);
			mwArray u_min(3, this->n_vertices(), mxDOUBLE_CLASS);
			mwArray curvature_max(1, this->n_vertices(), mxDOUBLE_CLASS);
			mwArray curvature_min(1, this->n_vertices(), mxDOUBLE_CLASS);
			mwArray normal(3, this->n_vertices(), mxDOUBLE_CLASS);
			mwArray hks(this->hks_len, this->n_vertices(), mxDOUBLE_CLASS);
			mwArray le(this->le_len * mul, this->n_vertices(), mxDOUBLE_CLASS);
			mwArray cf(1, (this->le_len * 3 + 1) * mul, mxDOUBLE_CLASS);
			mwArray filenameArray(this->filename.c_str());
			mwArray desc_len(1, 1, mxDOUBLE_CLASS);
			mwArray le_len(1, 1, mxDOUBLE_CLASS);
			mwArray diameter(1, 1, mxDOUBLE_CLASS);
			mwArray resp(3, this->n_vertices(), mxDOUBLE_CLASS);
			desc_len(1, 1) = this->hks_len;
			le_len(1, 1) = (this->le_len * 3.0 + 1) * mul / 3.0;
			mwArray cf_flag(1, 1, mxLOGICAL_CLASS);
			cf_flag(1, 1) = true;

			compute_curvature(10, u_max, u_min, curvature_max, curvature_min, normal, hks, diameter, resp, le, cf, vertices, faces, filenameArray, desc_len, le_len, cf_flag); //number output

			return cf;

		}


        bool is_vertex_info_computed() const
        {
            return is_vertex_info_computed_;
        }


        const VPropHandles &get_vphs() const
        {
            return vphs;
        }

		double get_le_len() const
		{
			return le_len;
		}

        double correct_calc_edge_length(EdgeHandle _eh) const
        {
            auto _heh = this->halfedge_handle(_eh, 0);
            auto  v = this->point(this->to_vertex_handle(_heh)) - this->point(this->from_vertex_handle(_heh));
            return v.length();
        }

        double mean_edge_length()
        {
            double sum = 0;
            for (auto &edge : this->edges())
            {
                sum += this->correct_calc_edge_length(edge);
            }

            return sum / this->n_edges();
        }


    };

} //end namespace GIGen

#endif //DGPC_MESH_H
