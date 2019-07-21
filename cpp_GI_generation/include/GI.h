/************************************************************
* This file is MODIFIED from a part of the 3D descriptor 
* learning framework by Hanyu Wang(王涵玉)
* https://github.com/jianweiguo/local3Ddescriptorlearning
*
* Author: Yiqun Wang(王逸群)
* https://github.com/yiqun-wang/LPS
************************************************************/

#ifndef GI_H
#define GI_H

#include <iostream>
#include <vector>
#include "GPC.h"
#include "Vector3.h"
#include "utils.h"

namespace GIGen
{
    //template <typename Mesh>
    typedef MeshOM Mesh;
    class GI
    {
    private:
        const GPC<Mesh> &gpc;
        const int gi_size, rotation_num, used_attr_num;
        std::vector<std::vector<std::vector<std::vector<float>>>> geo_img_all_rots; // Geometry images of all rotation angles.


        inline double triangle_area(const Point_2& p1, const Point_2& p2, const Point_2& p3)
        {
            double ax = p2.x() - p1.x();
            double ay = p2.y() - p1.y();
            double bx = p3.x() - p1.x();
            double by = p3.y() - p1.y();
            return fabs(0.5 * (ax * by - ay * bx));
        }


        bool append_features(const std::vector<std::vector<Point_2>>& sampling_points, const unsigned int &gi_idx, bool with_hks=false)
        {
            auto &gi = geo_img_all_rots[gi_idx];

            for (int r = 0; r < gi_size; r++)
            {
                for (int c = 0; c < gi_size; c++)
                {
                    auto &this_point = sampling_points[r][c];
                    auto idx_vec = gpc.find_triangle_vertices(this_point);

                    auto &&p0 = gpc.access_vertex_by_mesh_idx(idx_vec[0]);
                    auto &&p1 = gpc.access_vertex_by_mesh_idx(idx_vec[1]);
                    auto &&p2 = gpc.access_vertex_by_mesh_idx(idx_vec[2]);

                    double s0 = triangle_area(this_point, p1, p2);
                    double s1 = triangle_area(p0, this_point, p2);
                    double s2 = triangle_area(p0, p1, this_point);
                    double s = s0 + s1 + s2;

                    double p0_c_max, p1_c_max, p2_c_max;
                    double p0_c_min, p1_c_min, p2_c_min;
                    Point p0_normal, p1_normal, p2_normal;
					Point p0_resp, p1_resp, p2_resp;
					Point p0_point, p1_point, p2_point;

					std::vector<double> p0_le, p1_le, p2_le;
					p0.get_attr("le", p0_le);
					p1.get_attr("le", p1_le);
					p2.get_attr("le", p2_le);
					std::vector<double> p0_cf, p1_cf, p2_cf;
					p0.get_attr("cf", p0_cf);
					p1.get_attr("cf", p1_cf);
					p2.get_attr("cf", p2_cf);

					for (int i = 0; i < p0_cf.size(); i++)
					{
						gi[r][c].emplace_back((p0_cf[i] * s0 + p1_cf[i] * s1 + p2_cf[i] * s2) / s);
					}

					with_hks = false;
                    if (with_hks)
                    {
                        std::vector<double> p0_hks, p1_hks, p2_hks;
                        p0.get_attr("hks", p0_hks);
                        p1.get_attr("hks", p1_hks);
                        p2.get_attr("hks", p2_hks);

                        for (int i = 0; i < p0_hks.size(); i++)
                        {
                            gi[r][c].emplace_back((p0_hks[i] * s0 + p1_hks[i] * s1 + p2_hks[i] * s2) / s);
                        }
                    }

                }
            }

            return true;

        }



    public:

        GI(const GPC<Mesh> &gpc, const std::vector<double> &max_radius, const int &gi_size, const int &rotation_num, const int &used_attr_num = 5) :
            gpc(gpc),
            geo_img_all_rots(rotation_num, std::vector<std::vector<std::vector<float>>>(gi_size, std::vector<std::vector<float>>(gi_size))),
            gi_size(gi_size), rotation_num(rotation_num), used_attr_num(used_attr_num)
        {
            if (!gpc.point_num())
                return;

            // Initialization of the geometry image;
            for (auto &gi : this->geo_img_all_rots)
            {
                for (auto &c : gi)
                {
                    for (auto &p : c)
                    {
                        p.reserve(used_attr_num * max_radius.size());
                    }
                }
            }


            double start_x = -sqrt(2) / 2 + sqrt(2) / (2 * double(gi_size));
            double start_y = sqrt(2) / 2 - sqrt(2) / (2 * double(gi_size));
            double delta = sqrt(2) / (double(gi_size));


            //Sampling points
            std::vector<std::vector<Point_2>> generic_sampling_points(gi_size, std::vector<Point_2>(gi_size));
            for (int r = 0; r < gi_size; r++)
            {
                for (int c = 0; c < gi_size; c++)
                {
                    generic_sampling_points[r][c] = Point_2(start_x + c * delta, start_y - r * delta);
                }
            }



            double rotation_rad = 2 * M_PI / rotation_num;
            for (unsigned int i = 0; i < rotation_num; i++)
            {
                double rad = rotation_rad * i;

                for (double radius : max_radius)
                {
                    std::vector<std::vector<Point_2>> sampling_points = generic_sampling_points;

                    for (auto& row : sampling_points)
                    {
                        for (auto& point : row)
                        {
                            double x = point.x() * cos(rad) - point.y() * sin(rad); // Rotate the sampling points.
                            double y = point.x() * sin(rad) + point.y() * cos(rad);
                            point = Point_2(x * radius, y * radius); // Scale the sampling points to fit parameterization radius.
                        }
                    }

                    
                    if (radius == max_radius[max_radius.size() - 1])
                    {
                        this->append_features(sampling_points, i, true);
                    }
                    else
                    {
                        this->append_features(sampling_points, i, false);
                    }
                }


            }
        }


        bool save_all(const std::string &geo_img_dir, const std::string& name_prefix) const
        {
            return this->save_all(Dir(geo_img_dir), name_prefix);

        }

        bool save_all(const Dir &geo_img_dir, const std::string& name_prefix) const
        {
            for (unsigned int i = 0; i < this->rotation_num; i++)
            {
                auto geo_img_path = geo_img_dir + name_prefix + "_rot_" + to_string_f("%02d", i) + ".gi";

                std::ofstream out(geo_img_path);

                int count = 100;
                while (!out && count > 0)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    out = std::ofstream(geo_img_path);
                    count--;
                }
                if (!out) return false;

                for (int ch = 0; ch < geo_img_all_rots[i][0][0].size(); ch++)
                {
                    for (auto &r_vec : geo_img_all_rots[i])
                    {
                        for (auto &val : r_vec)
                            out << std::fixed << val[ch] << " ";
                        out << std::endl;
                    }
                    out << std::endl;
                }

                out.close();
            }
            return true;
        }

        bool save_all_rotation_in_one(const std::string &geo_img_dir, const std::string& name_prefix) const
        {
            auto geo_img_path = geo_img_dir + name_prefix + ".gi";

            std::ofstream out(geo_img_path);
            int count = 100;
            while (!out && count > 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                out = std::ofstream(geo_img_path);
                count--;
            }
            if (!out) return false;


			// 12*8*8*32-> 12*2*32*32
			for (unsigned int i = 0; i < this->rotation_num; i++)
			{
				int num = geo_img_all_rots[i][0][0].size() / 4;
				for (int ch = 0; ch < num; ch++)
				{
					for (auto &r_vec : geo_img_all_rots[i])
					{
						for (unsigned int n = 0; n < 4; n++)
						{
							for (auto &val : r_vec)
								out << std::fixed << val[4*ch+n] << " ";							
						}
						out << std::endl;
					}
					if (ch==num/2-1 || ch==num-1) { out << std::endl; }
				}

				out << std::endl << std::endl;

			}

            out.close();
            return true;
        }


    };


}; // End namespace GIGen


#endif // !GI_H
