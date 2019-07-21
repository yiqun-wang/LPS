/************************************************************
* Author: Hanyu Wang(王涵玉)
************************************************************/

#ifndef Path_H
#define Path_H

#include <cstdio>
#include <cctype>
#include <algorithm>
#include <string>
#include <ctime>
#include "dirent.h"


namespace GIGen
{
    class Path : public std::string
    {
    public:
        Path(const std::string &s)
        {
            this->assign(s);
        }

        Path() {}

        inline auto &pythonic_replace(const std::string& src, const std::string& dest) // Replace all src occured in *this into dest.
        {
            std::string::size_type pos = 0;
            while ((pos = this->find(src, pos)) != std::string::npos)
            {
                this->replace(pos, src.size(), dest);
                pos += dest.size();
            }
            return *this;
        }

        auto &operator=(const std::string &s) // Value assigning method.
        {
            return this->assign(s);
        }

    };


    class Dir : public Path
    {
    public:
        Dir(const std::string &s)
        {
            this->assign(s);
            if (this->size())
                this->append(this->at(this->size() - 1) != '/' ? "/" : ""); // use linux style path
        }
        Dir() {}

        bool ls_to_vector(std::vector<Path>& filenames) const // List folders/files in this dir into filenames.
        {
            const char *path = this->c_str();
            struct dirent* ent = NULL;
            DIR *pDir;
            pDir = opendir(path);

            if (pDir == NULL)
                return false;

            Path tmp;
            while (NULL != (ent = readdir(pDir)))
            {
                tmp = ent->d_name;
                if (tmp != "." && tmp != "..")
                    filenames.push_back(tmp);
            }
            return true;
        }

        //std::string &join(const std::string &p)
        //{
        //	return this->append(p);
        //}

        static Path join(const std::string &p0, const std::string &p1) // Join two path together (assuming the first one belongs to Dir).
        {
            return Path(Dir(p0).append(p1));
        }

    };


    template <typename _Printable>
    std::string to_string_f(const char* format, const _Printable p)
    {
        char buffer[20];
        sprintf(buffer, format, p);
        return std::string(buffer);
    }


    template <typename _ItemType>
    std::vector<_ItemType> read_vector(const std::string &filepath)
    {
        _ItemType a;
        std::vector<_ItemType> r_vec;
        std::ifstream file(filepath, std::ifstream::in);
        while (file >> a)
        {
            r_vec.emplace_back(a);
        }
        return r_vec;
    }


    bool is_useless_char(int ch)
    {
        return (ch == '[') || (ch == ']') || (ch == '{') || (ch == '}') || (ch == '(') || (ch == ')') || std::isspace(ch);
    }


    inline std::string &trim(std::string &s)
    {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) { return !is_useless_char(ch); }));
        s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { return !std::isspace(ch); }).base(), s.end());

        return s;
    }


    std::vector<double> parse_list(std::string s, const std::string &delimiter = ",")
    {
        trim(s);
        std::vector<double> r_vec;
        size_t pos = 0;
        std::string token;

        while ((pos = s.find(delimiter)) != std::string::npos)
        {
            token = s.substr(0, pos);
            r_vec.emplace_back(std::stod(token));
            s.erase(0, pos + delimiter.length());
        }

        if (trim(s) == "")
            return r_vec;
        r_vec.emplace_back(std::stod(s));

        return r_vec;
    }

    void show_progress_bar(const volatile int &current, const int &total, const time_t &begin_time)
    {
        const int total_progerss = 50;
        auto progress = total_progerss*(current) / total;
        std::cout << '\r' << "[" << std::string(progress, '#') << std::string(total_progerss - progress, ' ');
        std::cout << "]" << "    " << current << "/" << total;
        std::cout << "   total time cost: " << time(0) - begin_time << 's' << std::flush;
    }

} // End namespace GIGen

#endif // !Path_H