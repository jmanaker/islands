#pragma once
#include <numeric>
#include <utility>

#include "Tweaks.hpp"
#include "LinAlg3.hpp"

namespace islands
{
    namespace _internal
    {
        using containers::tree;
        constexpr auto const blank{ std::numeric_limits<tree::node>::max() };
    }

    template<typename data_t, std::size_t dim = 2>
    struct matrix
    {
        lin_alg3::slice<dim> coords;
        data_t data;
    };

    auto matrix_slice(std::size_t width, std::size_t height)
    {
        return lin_alg3::slice<2>{0, { {width, 1}, {height, static_cast<std::ptrdiff_t>(width)} }};
    }

    [[nodiscard]] std::size_t solve(matrix<bool const *> const &input)
    {
        using namespace _internal;
        //Forest of trees of island id
        //Island id is real if it "points" to itself; otherwise points to island coalesced with it
        //When two islands merge, we pick a root arbitrarily (the lesser) and 
        //make each root point there (so the lesser is real, b/c it points to itself)
        containers::tree known;
        //Avoid excess heap traffic by allocating outside the loop
        auto const &len{ input.coords.width().len };
        //Island ids for each point on the boundary between the studied and unstudied regions
        auto prevline{ std::make_unique_for_overwrite<tree::node[]>(len) }, //studied boundary
            curline{ std::make_unique_for_overwrite<tree::node[]>(len) }; //unstudied boundary
        //std::vector<id_key> prevline(len), //studied boundary
        //    curline(len); //unstudied boundary
        //Initially no islands in empty studied region
        std::uninitialized_fill_n(prevline.get(), len, blank);
        //We overwrite curline, so garbage data in there is OK
        for (std::size_t index{ 0 }; index < input.coords.height().len; ++index)
        {
            auto prev{ blank };
            for (std::size_t jndex{ 0 }; jndex < len; ++jndex)
            {
                if (!input.data[input.coords[{ jndex, index }].to_scalar()])
                    //Not a 1; irrelevant
                    prev = blank;
                else if (blank == prevline[jndex] && blank == prev)
                    //New island!
                    prev = known.add_new();
                else if (blank == prevline[jndex] && blank != prev)
                    //Infer from left
                    ;
                else if (blank != prevline[jndex] && blank == prev)
                    //Infer from top
                    prev = prevline[jndex];
                else
                    //blank != prevline[jndex] && blank != prev
                    //Hard case: islands coalesce
                    //Can reuse prev, but the meaning (id) of that island changes
                    known.coalesce(prev, prevline[jndex]);
                curline[jndex] = prev;
            }
            //We overwrite curline, so garbage data in there is OK
            std::swap(prevline, curline);
        }
        //std::cout <<
        //    "For " << data.width.len << "x" << data.height.len << " matrix, "
        //    "used " << size(known) << " id(s)." << std::endl;
        return known.count_roots();
    }

    namespace _internal
    {
        struct solve2
        {
            bool const *data;
        private:
            struct partial_soln
            {
                struct side
                {
                    lin_alg3::slice<> coords;
                    //std::unique_ptr<tree::node[]>
                    std::vector<tree::node> ids;
                } left, right;
                std::size_t ids_used{ 0 }, bulk_ct{ 0 };
            private:
                [[nodiscard]] auto boundary_ids(tree const &known)
                {
                    std::vector<tree::node> boundaries;
                    for (auto const bdry : { &left.ids, &right.ids })
                        for (auto &id : *bdry)
                            if (blank != id)
                                boundaries.emplace_back(id = known.trace_root(id));
                    return boundaries;
                }
                void shrink(auto const &lookup_table)
                {
                    for (auto const bdry : { &left.ids, &right.ids })
                        for (auto &id : *bdry)
                            if (blank != id)
                                id = lookup_table[id];
                }
            public:
                void normalize(tree const &known)
                {
                    //(1) Collect boundary ids
                    auto boundaries{ boundary_ids(known) };
                    //(2) Uniquify boundary ids
                    decltype(cbegin(boundaries)) const last{
                        utils::uniquify(begin(boundaries), end(boundaries))
                    };
                    //(3) Record number of ids used
                    ids_used = last - cbegin(boundaries);
                    //(4) Replace ids with order of appearance in boundaries
                    shrink(utils::invert_injection(cbegin(boundaries), last));
                }
                void reindex_above(tree::node bound)
                {
                    //Disjointify ids
                    for (auto const bdry : { &left.ids, &right.ids })
                    {
                        constexpr auto const overflow_is_safe{
                            std::is_unsigned_v<decltype(bound)> || std::is_unsigned_v<decltype(blank)>
                        };
                        if constexpr (overflow_is_safe)
                        {
                            //Double loop to enable vectorized codegen
                            for (auto &key : *bdry)
                                key += bound;
                            std::replace(begin(*bdry), end(*bdry), blank + bound, blank);
                        }
                        else
                            for (auto &key : *bdry)
                                if (blank != key)
                                    key += bound;
                    }
                }
            };

            void zip_boundaries(tree &known,
                partial_soln::side const &inner_left,
                partial_soln::side const &inner_right) const
            {
                //Zip inner boundary, coalescing islands along the way
                using std::cbegin;
                auto il{ cbegin(inner_left.coords) }, ir{ cbegin(inner_right.coords) };
                auto const &iri{ inner_right.ids };
                for (auto lr{ begin(inner_left.ids) }, rl{ begin(iri) }, stop{ end(iri) };
                    rl != stop;
                    ++lr, ++rl, ++il, ++ir)

                    if (il[data] && ir[data])
                        //Since we've already disjointified the left & right ids,
                        //*lr and *rl will always differ
                        known.coalesce_nocheck(*lr, *rl);
            }

            void check_sizes(partial_soln::side const &inner_left, partial_soln::side const &inner_right)
                const noexcept
            {
                auto const &ilc{ inner_left.coords }, &irc{ inner_right.coords };
                auto const &walk_data{ ilc.indices[0] };
                assert(walk_data == irc.indices[0]);
                assert(walk_data.len > 0);
                auto const shift{ irc.start - ilc.start };
                //If transposed matrix, condition at right fails
                assert(shift == 1 && walk_data.stride > 1 || shift == walk_data.len * walk_data.stride);
            }

            [[nodiscard]] partial_soln merge(partial_soln &&lhs, partial_soln &&rhs) const
            {
                auto const &inner_left{ lhs.right }, &inner_right{ rhs.left };
                check_sizes(inner_left, inner_right);
                rhs.reindex_above(lhs.ids_used);
                partial_soln retval{
                    std::move(lhs.left),
                    std::move(rhs.right),
                    lhs.ids_used + rhs.ids_used,
                    lhs.bulk_ct + rhs.bulk_ct
                };
                //Build id table
                tree known(retval.ids_used);
                zip_boundaries(known, inner_left, inner_right);
                //Assume everything goes into the bulk until proven otherwise
                retval.bulk_ct += known.count_roots();
                retval.normalize(known);
                retval.bulk_ct -= retval.ids_used;
                return retval;
            }

            inline [[nodiscard]] partial_soln analyze(lin_alg3::slice<> const &input) const
            {
                partial_soln retval{ { input, std::vector<tree::node>(input.indices[0].len) } };
                {
                    using std::begin;
                    auto prev{ blank };
                    auto data_iter{ begin(input) };
                    for (auto &id : retval.left.ids)
                    {
                        if (!data_iter[data])
                            prev = blank;
                        else if (blank == prev)
                            prev = retval.ids_used++;
                        id = prev;
                        data_iter++;
                    }
                }
                retval.right = retval.left;
                return retval;
            }
            public:
            [[nodiscard]] partial_soln operator()(lin_alg3::slice<2> const &coords) const
            {
                auto const &ht{ coords.height().len };
                assert(ht);
                if (ht > coords.width().len)
                    return (*this)(transpose(coords));
                if (1 == ht)
                    return analyze(coerce<1>(coords));
                else
                {
                    using lin_alg::all_t;
                    auto const split_pt{ ht / 2 };
                    lin_alg3::slice<> const left(0, { split_pt }), right(split_pt, { ht - split_pt });
                    return merge((*this)(coords[{all_t{}, left}]), (*this)(coords[{all_t{}, right}]));
                }
            }
        };
    }

    [[nodiscard]] std::size_t solve2(matrix<bool const *> const &input)
    {
        auto const &retval{ _internal::solve2{input.data}(input.coords) };
        return retval.bulk_ct + retval.ids_used;
    }

    namespace _internal
    {
        [[nodiscard]] auto emplace_zeros(matrix<bool const *> const &input)
        {
            auto const &mat{ input.coords };
            auto const &ht{ mat.height().len }, &wt{ mat.width().len };
            auto const new_width{ wt + 2 }, new_height{ht + 2};
            matrix<std::unique_ptr<bool[]>> retval{
                matrix_slice(new_width, new_height), std::make_unique<bool[]>(new_width * new_height)
            };
            for (std::size_t index{ 0 }; index < wt; ++index)
                for (std::size_t jndex{ 0 }; jndex < ht; ++jndex)
                    retval.data[retval.coords[{index + 1, jndex + 1}].to_scalar()] =
                        input.data[mat[{index, jndex}].to_scalar()];
            return retval;
        }

        struct solve3 : matrix<std::unique_ptr<bool[]>>
        {
            void swarm_launch(std::size_t x, std::size_t y)
            {
                if (auto & cur_loc{ data[coords[{x, y}].to_scalar()] })
                {
                    cur_loc = false;
                    std::ptrdiff_t dx, dy;
                    for (auto const coord : {&dx, &dy})
                        for (auto const dir : {-1, 1})
                        {
                            dx = dy = 0;
                            *coord = dir;
                            swarm_launch(x + dx, y + dy);
                        }
                }
            }
            [[nodiscard]] auto operator()(void)
            {
                std::size_t count{ 0 };
                for (std::size_t index{ 0 }; index < coords.width().len; ++index)
                    for (std::size_t jndex{ 0 }; jndex < coords.height().len; ++jndex)
                        if (data[coords[{index, jndex}].to_scalar()])
                        {
                            ++count;
                            swarm_launch(index, jndex);
                        }
                return count;
            }
        };
    }

    [[nodiscard]] std::size_t solve3(matrix<bool const *> const &input)
    {
        return _internal::solve3{_internal::emplace_zeros(input)}();
    }
}