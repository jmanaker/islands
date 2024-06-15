//Inspired by https://codereview.stackexchange.com/questions/191747/c-interview-problem-finding-an-island-in-a-2d-grid/
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "Tweaks.hpp"

#include "LinAlg3.hpp"
#include "Islands.hpp"

#define GIANT
//#define FUZZ_FIXED_SIZE

enum intbool : bool;

namespace test
{
    std::default_random_engine engine(std::random_device{}()); //Use random seed, but after that anything's OK
    std::geometric_distribution<unsigned int> sizing(0.001); //Prob is p <=> mean is 1/p-1
    std::discrete_distribution<> coin_flip({ 0.501, 0.499 }); //2D percolation is critical at 0.5

    namespace internal
    {
        auto &print_line(islands::matrix<bool const *, 1> const &line, std::ostream &dest, 
            char left = '|', char right = '|')
        {
            dest << left;
            utils::scoped_streamstate<std::ios_base::boolalpha> _guard{ dest };
            dest << std::noboolalpha;
            for (auto const &index : line.coords)
                dest << line.data[index];
            return dest << right;
        }
    }

    auto &print_matrix(islands::matrix<bool const *> const &mat, std::ostream &dest)
    {
        using lin_alg::all_t;
        switch (mat.coords.height().len)
        {
        case 0: return dest << "[]";
        case 1: return internal::print_line({ coerce<1>(mat.coords), mat.data }, dest, '[', ']');
        default:
            auto start{ mat.coords.rows_begin() }, stop{ mat.coords.rows_end() };
            --stop;
            internal::print_line({ *start++, mat.data }, dest, '/', '\\') << std::endl;
            while (start < stop)
                internal::print_line({ *start++, mat.data }, dest) << std::endl;
            return
                internal::print_line({ *stop++, mat.data }, dest, '\\', '/');
        }
    }

    //Can't use vector<bool> b/c that has no underlying array to access
    auto resize_matrix(std::size_t const wt, std::size_t ht,
        islands::matrix<std::vector<intbool>> &giant)
    {
        std::size_t const sz{ wt * ht };
        giant.data.resize(sz);
        giant.coords = islands::matrix_slice(wt, ht);
        return sz;
    }

    template<bool always_print = true>
    bool analyze(islands::matrix<std::vector<intbool>> &giant, std::size_t const sz)
    {
        for (std::size_t index{ 0 }; index < sz; ++index)
            giant.data[index] = intbool(coin_flip(engine));
        std::size_t value, value2, value3;
        {
            struct loop_iter_t
            {
                char const *const txt;
                size_t(*func)(islands::matrix<bool const *> const &);
                std::size_t &dest;
            } iters[]{
                {"V1: ", &islands::solve, value},
                {"V2: ", &islands::solve2, value2},
                {"V3: ", &islands::solve3, value3},
            };
            for (auto const &iter : iters)
            {
                //Can't use if constexpr b/c that creates a new scope, 
                //but a pair of reference/ptrs gets optimized out
                std::conditional_t<always_print,
                    timer,
                    std::pair<std::ostream&, char const*const> const> 
                    _guard(std::cout, iter.txt);
                iter.dest = (*iter.func)({ giant.coords, reinterpret_cast<bool*>(giant.data.data()) });
            }
        }
        auto retval{ value != value2 || value2 != value3 };
        if (always_print || retval)
        {
            if (giant.coords.height().len < 100 && giant.coords.width().len < 100)
            {
                std::cout << "Giant matrix is:" << std::endl;
                test::print_matrix({ giant.coords, reinterpret_cast<bool *>(giant.data.data()) },
                    std::cout);
                std::cout << std::endl;
            }
            std::cout <<
                "(V1) " << value << " (V2) " << value2 << " (V3) " << value3 << 
                " islands in giant matrix." << std::endl;
        }
        return retval;
    }

    namespace _internal
    {
        template<std::size_t count>
        struct nested_iter abstract final
        {
            typedef std::initializer_list<typename nested_iter<count - 1>::type> type;
        };
        template<> struct nested_iter<0> abstract final { typedef bool type; };
    }

    template<std::size_t count> using nested_iter_t = _internal::nested_iter<count>::type;

    struct matrix_view : islands::matrix<std::unique_ptr<bool[]>>
    {
    private:
        static constexpr std::size_t const dim{ 2 };
        typedef lin_alg3::slice_axis index;
        template<typename il>
        constexpr index *init_sizes(std::initializer_list<il> const &data, index *idx)
        {
            using std::cend;
            assert(idx < cend(coords.indices));
            using std::size;
            idx->len = std::max(size(data), idx->len);
            if constexpr (!std::is_same_v<il, bool>)
            {
                index *retval{ nullptr };
                assert(size(data));
                for (auto const &list : data)
                    retval = init_sizes(list, idx - 1);
                return retval;
            }
            return idx;
        }
        template<typename il>
        constexpr auto copy_data(std::initializer_list<il> const &data, index *idx, bool *loc)
        {
            using std::begin, std::end;
            if constexpr (std::is_same_v<il, bool>)
                std::move(begin(data), end(data), loc);
            else
            {
                auto loc_cpy{ loc };
                for (auto const &list : data)
                {
                    copy_data(list, idx - 1, loc_cpy);
                    loc_cpy += idx->stride;
                }
            }
        }
    public:
        matrix_view(nested_iter_t<dim> data)
        {
            static_assert(dim > 0);
            for (auto &obj : coords.indices)
                obj.len = 0;
            using std::begin, std::cbegin, std::end, std::cend;
            auto loc{ init_sizes(data, end(this->coords.indices) - 1) };
            assert(cbegin(coords.indices) == loc);
            loc->stride = 1;
            std::size_t size(loc->stride);
            {
                auto const stop{ cend(coords.indices) };
                decltype(loc) prev;
                while ((prev = loc++) != stop)
                    size *= (loc->stride = prev->len);
            }
            this->data = std::make_unique<bool[]>(size);
            copy_data(data, this->coords.indices + dim - 1, this->data.get());
        }
    };
    auto operator<=(std::size_t result, matrix_view input)
    {
        return std::make_pair(std::move(input), std::move(result));
    }
    static std::pair<matrix_view, std::size_t> test_cases[]{
        6 <= matrix_view{
            {0,    1,    0,    1,    0},
            {0,    0,    1,    1,    1},
            {1,    0,    0,    1,    0},
            {0,    1,    1,    0,    0},
            {1,    0,    1,    0,    1}
        },
        0 <= matrix_view{{ 0 }},
        1 <= matrix_view{{ 1 }},
        2 <= matrix_view{
            { 1, 0, 1, 0 }
        },
        2 <= matrix_view{
            { 1, 0, 1, 0 },
            { 0, 1, 1, 1 },
            { 0, 0, 1, 0 }
        },
        4 <= matrix_view{
            {1,0,1,0},
            {0,1,1,1},
            {0,0,1,0},
            {1,1,0,0},
            {0,1,0,1}
        },
        5 <= matrix_view{
            {0,1,0,1,0},
            {0,0,1,1,1},
            {1,0,1,1,0},
            {0,1,1,0,0},
            {1,0,1,0,1}
        },
        1 <= matrix_view{
            {1,1,1,1,1},
            {1,1,1,1,1},
            {1,1,1,1,1},
            {1,1,1,1,1},
            {1,1,1,1,1}
        },
        12 <= matrix_view{
            /*
            {C,*,*,D,D,D,*,*,F,F,*,D},
            {C,*,D,D,D,*,*,E,*,*,D,D},
            {*,D,D,D,*,D,*,*,D,D,D,D},
            {*,*,*,D,D,D,D,D,*,D,*,D},
            {*,G,G,*,*,D,*,*,*,D,D,D},
            {G,G,G,G,*,D,D,*,L,*,D,D},
            {G,G,*,*,*,*,D,*,*,D,D,*},
            {G,*,*,*,*,*,D,D,D,D,D,*},
            {*,H,*,*,*,*,D,D,D,D,*,B},
            {H,H,*,*,J,*,D,*,D,*,*,B},
            {*,*,I,*,*,K,*,*,*,*,B,B},
            {*,*,I,*,*,K,K,*,A,A,*,*}*/

            {1,0,0,1,1,1,0,0,1,1,0,1},
            {1,0,1,1,1,0,0,1,0,0,1,1},
            {0,1,1,1,0,1,0,0,1,1,1,1},
            {0,0,0,1,1,1,1,1,0,1,0,1},
            {0,1,1,0,0,1,0,0,0,1,1,1},
            {1,1,1,1,0,1,1,0,1,0,1,1},
            {1,1,0,0,0,0,1,0,0,1,1,0},
            {1,0,0,0,0,0,1,1,1,1,1,0},
            {0,1,0,0,0,0,1,1,1,1,0,1},
            {1,1,0,0,1,0,1,0,1,0,0,1},
            {0,0,1,0,0,1,0,0,0,0,1,1},
            {0,0,1,0,0,1,1,0,1,1,0,0}
        },
        11 <= matrix_view{
            /*
            {*,*,*,A,A,A,A,*,B,B,*,C},
            {*,*,D,*,A,A,*,*,B,*,*,C},
            {A,A,*,*,A,A,*,*,B,B,*,*},
            {*,A,A,A,A,*,*,E,*,B,*,*},
            {*,*,A,A,A,*,F,*,*,*,A,A},
            {*,G,*,A,A,A,*,A,A,A,A,A},
            {A,*,A,A,*,*,*,A,A,*,A,A},
            {A,*,*,A,A,A,A,A,*,A,A,*},
            {A,A,A,A,A,A,A,*,H,*,A,*},
            {*,*,*,A,A,A,*,H,H,*,A,A},
            {I,*,A,A,*,*,*,H,*,J,*,A},
            {*,K,*,A,A,*,H,H,H,*,*,A}
            */

            {0,0,0,1,1,1,1,0,1,1,0,1},
            {0,0,1,0,1,1,0,0,1,0,0,1},
            {1,1,0,0,1,1,0,0,1,1,0,0},
            {0,1,1,1,1,0,0,1,0,1,0,0},
            {0,0,1,1,1,0,1,0,0,0,1,1},
            {0,1,0,1,1,1,0,1,1,1,1,1},
            {1,0,1,1,0,0,0,1,1,0,1,1},
            {1,0,0,1,1,1,1,1,0,1,1,0},
            {1,1,1,1,1,1,1,0,1,0,1,0},
            {0,0,0,1,1,1,0,1,1,0,1,1},
            {1,0,1,1,0,0,0,1,0,1,0,1},
            {0,1,0,1,1,0,1,1,1,0,0,1}
        },
        20 <= matrix_view{
            /*
            {A,A,*,*,*,B,B,B,B,*,*,*,C,C,C,C},
            {A,A,A,*,*,*,*,*,*,D,*,*,*,*,*,C},
            {A,*,*,*,*,E,E,E,*,*,*,F,F,*,*,*},
            {A,*,G,*,*,E,*,*,H,H,H,*,F,F,F,*},
            {A,*,G,*,*,*,I,*,*,*,H,*,F,F,F,F},
            {*,*,*,J,*,*,I,I,*,*,H,*,*,F,*,F},
            {J,*,*,J,J,J,*,*,*,*,H,*,T,*,*,F},
            {J,J,J,J,J,J,*,*,K,K,*,*,*,*,L,*},
            {*,*,*,*,*,J,J,*,K,*,M,M,*,L,L,L},
            {*,*,*,J,J,J,*,M,*,*,M,*,*,*,*,*},
            {*,*,J,J,*,*,*,M,M,M,M,M,M,M,M,*},
            {N,N,*,*,O,*,*,M,M,*,*,*,M,M,*,P},
            {*,*,Q,Q,*,M,M,*,M,M,M,*,M,*,P,P},
            {R,*,*,*,M,*,M,M,M,M,M,*,M,*,*,P},
            {R,R,*,M,M,M,M,*,*,M,*,*,M,*,*,*},
            {R,*,*,*,*,M,*,M,M,M,*,*,M,*,*,M},
            {*,*,S,S,*,M,M,M,*,*,M,M,M,M,M,M}
            */
            {1,1,0,0,0,1,1,1,1,0,0,0,1,1,1,1},
            {1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1},
            {1,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0},
            {1,0,1,0,0,1,0,0,1,1,1,0,1,1,1,0},
            {1,0,1,0,0,0,1,0,0,0,1,0,1,1,1,1},
            {0,0,0,1,0,0,1,1,0,0,1,0,0,1,0,1},
            {1,0,0,1,1,1,0,0,0,0,1,0,1,0,0,1},
            {1,1,1,1,1,1,0,0,1,1,0,0,0,0,1,0},
            {0,0,0,0,0,1,1,0,1,0,1,1,0,1,1,1},
            {0,0,0,1,1,1,0,1,0,0,1,0,0,0,0,0},
            {0,0,1,1,0,0,0,1,1,1,1,1,1,1,1,0},
            {1,1,0,0,1,0,0,1,1,0,0,0,1,1,0,1},
            {0,0,1,1,0,1,1,0,1,1,1,0,1,0,1,1},
            {1,0,0,0,1,0,1,1,1,1,1,0,1,0,0,1},
            {1,1,0,1,1,1,1,0,0,1,0,0,1,0,0,0},
            {1,0,0,0,0,1,0,1,1,1,0,0,1,0,0,1},
            {0,0,1,1,0,1,1,1,0,0,1,1,1,1,1,1}
        }
    };
}

int _cdecl main()
{
    using namespace test;
    for (auto const &test_case : test_cases)
    {
        //Run first for exception-safety
        auto const &input{ test_case.first };
        islands::matrix<bool const *> const converted{ input.coords, input.data.get() };
        auto const value{ islands::solve(converted) },
            value2{ islands::solve2(converted) },
            value3{ islands::solve3(converted) };
        std::cout <<
            "Expected " << std::setw(2) << test_case.second << ", got "
            "(V1) " << std::setw(2) << value << " "
            "(V2) " << std::setw(2) << value2 << " "
            "(V3) " << std::setw(3) << value3 << std::endl;
    }
    islands::matrix<std::vector<intbool>> giant;
    std::size_t wt, ht;
#ifndef GIANT
    wt = sizing(engine), ht = sizing(engine);
    auto const sz{ resize_matrix(wt, ht, giant) };
    analyze<>(giant, sz);
#else
    try
    {
#ifdef FUZZ_FIXED_SIZE
        wt = 8, ht = 16;
        std::size_t const sz{ resize_matrix(wt, ht, giant) };
        for (std::size_t count{ 0 };;)
        {
            for (; count % 1024; ++count)
                if (analyze<false>(giant, sz))
                    return 0;
            std::cout << "Tested " << count << "; no crash yet..." << std::endl;
        }
#else
        for (std::size_t sum{ 2 }; sum < /*200*/100; ++sum)
        {
            for (wt = 1; wt < std::min<std::size_t>(sum, 100); ++wt)
            {
                ht = sum - wt;
                auto const sz{ resize_matrix(wt, ht, giant) };
                for (std::size_t count{ 0 }; count < 1024; ++count)
                    if (analyze<false>(giant, sz))
                        return 0;
            }
            std::cout << "Sum = " << sum << "; no errors found." << std::endl;
        }
#endif
    }
    catch (...)
    {
        std::cout
            << "ERROR! ERROR! ERROR!" << std::endl
            << "Giant matrix is:" << std::endl;
        test::print_matrix({ giant.coords, reinterpret_cast<bool *>(giant.data.data()) }, std::cout);
        std::cout << std::endl;
    }
#endif
    return 0;
}