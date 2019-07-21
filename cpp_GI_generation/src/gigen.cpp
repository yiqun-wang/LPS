/************************************************************
* This file is MODIFIED from a part of the 3D descriptor 
* learning framework by Hanyu Wang(王涵玉)
* https://github.com/jianweiguo/local3Ddescriptorlearning
*
* Author: Yiqun Wang(王逸群)
* https://github.com/yiqun-wang/LPS
************************************************************/

#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif // !_CRT_SECURE_NO_WARNINGS
#endif

#include <iostream>
#include "Generator.h"
#include "Mesh_C.h"
#include "GPC.h"
#include "utils.h"
#include <GI.h>
#include <limits>
#include <ctime>
#include <thread>
#include <direct.h>
#include <io.h>
#include <mutex>
#include "dirent.h"
#include "IniFile.h"
#include "libcompcur.h"
#include "mclmcr.h"
#include "matrix.h"
#include "mclcppclass.h"


using namespace OpenMesh;
using namespace GIGen;
using namespace std;


// mutex to control global_finished_count_ptr
static mutex mtx;

void usage_error(const char* progname)
{
    cout << "Usage: " << progname << " config_filename " << endl;
    cout << endl;
    cout << "Examples: " << endl;
    cout << endl;
    cout << "  " << "Compute geometry images using config.ini" << endl;
    cout << "  " << progname << " config.ini" << endl;
    cout << endl;
    exit(-1);
}


// Geometry image generation function
void process_one_point(Mesh &curr_mesh,
    const unsigned int &source_idx_,
    const double &max_r,
    const std::vector<double> &radius_list,
    const int &gi_size,
    const int &rotation_num,
    const Dir &gi_dir,
    const std::string& name_prefix,
    volatile int *global_finished_count_ptr)
{
    GPC<Mesh> gpc(curr_mesh, source_idx_, max_r, radius_list);
	gpc.compute_GPC();
    GI gi(gpc, radius_list, gi_size, rotation_num);

    if (!gi.save_all_rotation_in_one(gi_dir, name_prefix))
    {
        cerr << "Failed to save geometry images." << endl;
        cerr << gi_dir + name_prefix + "_rot_x.gi" << endl;
    }

    while (!mtx.try_lock());
    (*global_finished_count_ptr)++;
    mtx.unlock();

}


int main(int argc, char** argv)
{
    time_t begin_time = time(0);

    // Parse options.
    if (argc != 2) usage_error(argv[0]);
    const std::string& config_filepath = argv[1];

    // Initialize MATLAB Computation Library.
    if (!libcompcurInitialize())
    {
        cerr << "Could not initialize libcompcur!" << endl;
        exit(-1);
    }

    cout << "The libcompcur Initialization Success!" << endl;


    // Get configurations from config file.
    bool got_kpis = false;
    bool got_radius_list = false;
    vector<int> kpi_set;
    vector<double> radius_list_p, radius_list;

    IniFile ini(config_filepath);

    ini.setSection("dirs");
    auto mesh_dir_str = ini.readStr("mesh_dir");
    auto gi_dir_str = ini.readStr("gi_dir");
    auto kpi_dir_str = ini.readStr("kpi_dir");

    ini.setSection("settings");
    int gi_size = ini.readInt("gi_size");
    int hks_len = ini.readInt("hks_len");
    int rotation_num = ini.readInt("rotation_num");
    auto radius_list_str = ini.readStr("radius_list_p");

    if (radius_list_str == "default")
    {
		radius_list_p = { 0.021, 0.028, 0.035 };
    }
    else
    {
        radius_list_p = parse_list(radius_list_str);
    }

    auto max_r_iter = max_element(radius_list_p.begin(), radius_list_p.end());
    double max_r = (*max_r_iter) * 1.33333333;

    if (ini.readStr("using_all_points") != "true")
    {
        kpi_set = read_vector<int>(kpi_dir_str);
        if (!kpi_set.size())
        {
            cerr << "Cannot read keypoint indices!" << endl;
            exit(-1);
        }
        got_kpis = true;
    }


    const Dir mesh_dir(mesh_dir_str);
    const Dir gi_dir(gi_dir_str);

    vector<Path> filenames, off_filenames;
    if (!mesh_dir.ls_to_vector(filenames))
    {
        std::cerr << "Fail to list files." << std::endl;
        return 1;
    }

    if (!(_access(gi_dir.c_str(), 00/*check if it exists*/) == 0)/*not exist!*/)
    {
        if (_mkdir(gi_dir.c_str()) == -1)
        {
            cerr << "Failed to make directory." << endl;
            exit(-1);
        }
    }

    // Extract off files.
    for (const auto &filename : filenames)
    {
        std::string&& ext = filename.substr(filename.rfind('.') == std::string::npos ? filename.length() : filename.rfind('.') + 1);
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == "off")
        {
            off_filenames.emplace_back(filename);
        }
    }


	for (size_t filename_i = 0; filename_i < off_filenames.size(); filename_i++)
	{
		const auto &filename = off_filenames[filename_i];
		cout << "Processing " << filename << ". " << filename_i + 1 << "/" << off_filenames.size() << endl;

		auto name_prefix_base = Path(filename).pythonic_replace(".off", "");
		auto &&fullpath = Dir::join(mesh_dir, filename);
		Mesh curr_mesh(fullpath);

		//if (!got_kpis)
		{
			int n = curr_mesh.n_vertices();
			kpi_set.resize(n);
			for (size_t i = 0; i < n; i++)
			{
				kpi_set[i] = i;
			}
			got_kpis = true;
		}

		//modified list
		//if (!got_radius_list)
		{
			double tmp = curr_mesh.diameter;
			radius_list.clear();
			for (auto &i : radius_list_p)
			{
				radius_list.emplace_back(i * tmp);
			}
			max_r = (*max_r_iter) * 1.33333333;
			max_r *= tmp;

			got_radius_list = true;
			//continue;
		}


		auto curr_gi_dir = gi_dir + Dir(name_prefix_base);

		if (!(_access(curr_gi_dir.c_str(), 00/*check if it exists*/) == 0)/*not exist!*/)
		{
			if (mkdir(curr_gi_dir.c_str()) == -1)
			{
				cerr << "Failed to make directory." << endl;
				exit(-1);
			}
		}

		////////compute_cf
		for (size_t i = 0; i < /*kpi_n*/ curr_mesh.n_vertices(); i++)
		{
			GPC<Mesh> gpc(curr_mesh, /*kpi_set[i]*/ i, max_r, radius_list);
			gpc.compute_cf(i, name_prefix_base);
			std::cout << "iterations " << i + 1 << std::endl;
		}


		////////compute_cf45 for keypoint
		//size_t kpi_n = kpi_set.size();
		//for (size_t i = 0; i < /*kpi_n*/ curr_mesh.n_vertices(); i++)
		//{
		//	GPC<Mesh> gpc(curr_mesh, /*kpi_set[i]*/ i, max_r, radius_list);
		//	gpc.compute_cf_45(/*kpi_set[i]*/ i, name_prefix_base);
		//	std::cout << "iterations " << i + 1 << std::endl;
		//}


        size_t kpi_num = kpi_set.size();

        // global_finished_count is shared by threads
        volatile int global_finished_count = 0;
        volatile int *global_finished_count_ptr = &global_finished_count;
        for (size_t i = 0; i < kpi_num; i++)
        {
            std::string name_prefix = name_prefix_base + "_pidx_" + to_string_f("%04d", i);



            auto geo_img_path = curr_gi_dir + name_prefix + ".gi";

            if (_access(geo_img_path.c_str(), 00/*check if it exists*/) == 0 /*exist!*/)
            {
                while (!mtx.try_lock());
                global_finished_count++;
                mtx.unlock();
                //fin.close();
                continue;
            }

            // Multithread geometry image generation.
            thread t(process_one_point, curr_mesh, kpi_set[i], max_r, radius_list, gi_size, rotation_num, curr_gi_dir, name_prefix, global_finished_count_ptr);
            t.detach();

            // Single thread version
            //process_one_point(curr_mesh, kpi_set[i], max_r, radius_list, gi_size, rotation_num, curr_gi_dir, name_prefix, global_finished_count_ptr);

            show_progress_bar(global_finished_count, kpi_num, begin_time);

        }

        // Waiting for all threads to finish. 
        while (global_finished_count < kpi_num)
        {
            cout << "waiting... finished count: " << global_finished_count << std::flush;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        show_progress_bar(global_finished_count, kpi_num, begin_time);

        cout << endl;

    }

    libcompcurTerminate();
    return 0;
}


