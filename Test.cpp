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

#include "LinAlg.hpp"
#include "Islands.hpp"

#define GIANT
//#define FUZZ_FIXED_SIZE

namespace test
{
    [[nodiscard]] auto operator<=(std::size_t result, lin_alg::matrix_view input)
    {
        return std::make_pair(std::move(input), std::move(result));
    }
    std::default_random_engine engine(std::random_device{}()); //Use random seed, but after that anything's OK
    std::geometric_distribution<unsigned int> sizing(0.001); //Prob is p <=> mean is 1/p-1
    std::discrete_distribution<> coin_flip({ 0.501, 0.499 }); //2D percolation is critical at 0.5

    namespace internal
    {
        auto &print_line(lin_alg::vector const &line, std::ostream &dest, char left = '|', char right = '|')
        {
            dest << left;
            for (auto const &elem : line)
                dest << elem ? '1' : '0';
            return dest << right;
        }
    }

    auto &print_matrix(lin_alg::matrix_view const &mat, std::ostream &dest)
    {
        using lin_alg::all_t;
        switch (mat.height().len)
        {
        case 0: return dest << "[]";
        case 1: return internal::print_line(lin_alg::vector{ mat[{all_t{}, 0}] }, dest, '[', ']');
        default:
            std::size_t index{ 0 };
            internal::print_line(lin_alg::vector{ mat[{all_t{}, index++}] }, dest, '/', '\\');
            while (index < mat.height().len - 1)
                internal::print_line(lin_alg::vector{ mat[{all_t{}, index++}] }, dest << std::endl);
            return
                internal::print_line(lin_alg::vector{ mat[{all_t{}, index++}] }, dest << std::endl,
                    '\\', '/');
        }
    }

    auto resize_matrix(std::size_t const wt, std::size_t ht, lin_alg::matrix_view &giant)
    {
        std::size_t const sz{ wt * ht };
        giant.backing = std::make_shared_for_overwrite<lin_alg::data_t[]>(sz);
        giant.height().stride = giant.width().len = wt;
        giant.height().len = ht;
        return sz;
    }

    template<bool always_print = true>
    bool analyze(lin_alg::matrix_view &giant, std::size_t const sz)
    {
        for (std::size_t index{ 0 }; index < sz; ++index)
            giant.backing[index] = coin_flip(engine);
        std::size_t value, value2, value3;
        {
            struct loop_iter_t
            {
                char const *const txt;
                size_t(*func)(lin_alg::matrix_view const &);
                std::size_t &dest;
            } iters[]{
                {"V1: ", &islands::solve, value},
                {"V2: ", &islands::solve2, value2},
                {"V3: ", &islands::solve3, value3},
            };
            for (auto const &iter : iters)
            {
                std::conditional_t<always_print, timer, std::pair<std::ostream&, char const*const> const> 
                    _guard(std::cout, iter.txt);
                iter.dest = (*iter.func)(giant);
            }
        }
        auto retval{ value != value2 || value2 != value3 };
        if (always_print || retval)
        {
            if (giant.height().len < 100 && giant.width().len < 100)
            {
                std::cout << "Giant matrix is:" << std::endl;
                test::print_matrix(giant, std::cout);
                std::cout << std::endl;
            }
            std::cout <<
                "(V1) " << value << " (V2) " << value2 << " (V3) " << value3 << 
                " islands in giant matrix." << std::endl;
        }
        return retval;
    }

    using lin_alg::matrix_view;
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
    using lin_alg::matrix_view;
    for (auto &test_case : test_cases)
    {
        //Run first for exception-safety
        auto const value{ islands::solve(test_case.first) },
            value2{ islands::solve2(test_case.first) },
            value3{ islands::solve3(test_case.first) };
        std::cout <<
            "Expected " << std::setw(2) << test_case.second << ", got "
            "(V1) " << std::setw(2) << value << " "
            "(V2) " << std::setw(2) << value2 << " "
            "(V3) " << std::setw(3) << value3 << std::endl;
    }
    matrix_view giant;
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
        for (std::size_t sum{ 2 }; sum < /*200*/50; ++sum)
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
        test::print_matrix(giant, std::cout);
        std::cout << std::endl;
    }
#endif
    return 0;
}