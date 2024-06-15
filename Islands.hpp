#pragma once
#include <numeric>
#include <utility>

#include "Tweaks.hpp"
#include "LinAlg.hpp"

namespace islands
{
    namespace _internal
    {
        using containers::tree;
        constexpr auto const blank{ std::numeric_limits<tree::node>::max() };
    }

    [[nodiscard]] std::size_t solve(lin_alg::matrix_view const &data)
    {
        using namespace _internal;
        assert(!data.width().start && !data.height().start);
        //Forest of trees of island id
        //Island id is real if it "points" to itself; otherwise points to island coalesced with it
        //When two islands merge, we pick a root arbitrarily (the lesser) and 
        //make each root point there (so the lesser is real, b/c it points to itself)
        containers::tree known;
        //Avoid excess heap traffic by allocating outside the loop
        auto const &len{ data.width().len };
        //Island ids for each point on the boundary between the studied and unstudied regions
        auto prevline{ std::make_unique_for_overwrite<tree::node[]>(len) }, //studied boundary
            curline{ std::make_unique_for_overwrite<tree::node[]>(len) }; //unstudied boundary
        //std::vector<id_key> prevline(len), //studied boundary
        //    curline(len); //unstudied boundary
        //Initially no islands in empty studied region
        std::uninitialized_fill_n(prevline.get(), len, blank);
        //We overwrite curline, so garbage data in there is OK
        for (std::size_t index{ 0 }; index < data.height().len; ++index)
        {
            auto prev{ blank };
            for (std::size_t jndex{ 0 }; jndex < len; ++jndex)
            {
                if (!data[{jndex, index}])
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
        struct partial_soln
        {
        private:
            [[nodiscard]] auto boundary_ids(tree const &known)
            {
                std::vector<tree::node> boundaries;
                for (auto const bdry : { &left, &right })
                    for (auto &id : *bdry)
                        if (blank != id)
                            boundaries.emplace_back(id = known.trace_root(id));
                return boundaries;
            }
            void shrink(auto const &lookup_table)
            {
                for (auto const bdry : { &left, &right })
                    for (auto &id : *bdry)
                        if (blank != id)
                            id = lookup_table[id];
            }
        public:
            lin_alg::matrix_view region;
            //std::unique_ptr<id_key[]> left, right;
            std::vector<tree::node> left, right;
            std::size_t ids_used{ 0 }, bulk_ct{ 0 };
            void normalize(tree const &known)
            {
                //(1) Collect boundary ids
                auto boundaries{ boundary_ids(known) };
                //(2) Uniquify boundary ids
                decltype(cbegin(boundaries)) const last {
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
                for (auto const bdry : { &left, &right })
                {
                    //Double loop to enable vectorized codegen
                    for (auto &key : *bdry)
                        key += bound;
                    std::replace(begin(*bdry), end(*bdry), blank + bound, blank);
                }
            }
        };

        [[nodiscard]] partial_soln operator|(partial_soln left, partial_soln right)
        {
            assert(left.region.backing == right.region.backing);
            assert(left.region.width() == right.region.width());
            auto const &l_indices{ left.region.height() }, &r_indices{ right.region.height() };
            assert(l_indices.stride == r_indices.stride);
            assert(r_indices.start - l_indices.start == l_indices.len * l_indices.stride);
            assert(l_indices.len > 0 && r_indices.len > 0);

            right.reindex_above(left.ids_used);

            partial_soln retval
            {
                lin_alg::matrix_view{
                    left.region.backing, left.region.width(),
                    lin_alg::index{l_indices.start, l_indices.len + r_indices.len, l_indices.stride}
                },
                std::move(left.left),
                std::move(right.right),
                left.ids_used + right.ids_used,
                left.bulk_ct + right.bulk_ct
            };

            //Build id table
            tree known(retval.ids_used);
            {
                using lin_alg::all_t;
                //Zip inner boundary, coalescing islands along the way
                lin_alg::vector inner_left{ retval.region[{all_t{}, l_indices.len - 1}] },
                    inner_right{ retval.region[{all_t{}, l_indices.len}] };
                auto il{ begin(inner_left) }, ir{ begin(inner_right) };
                for (auto lr{ begin(left.right) }, rl{ begin(right.left) }, stop{ end(right.left) };
                    rl != stop;
                    ++lr, ++rl, ++il, ++ir)

                    if (*il && *ir)
                        //Since we've already disjointified the left & right ids,
                        //*lr and *rl will always differ
                        known.coalesce_nocheck(*lr, *rl);
            }
            retval.bulk_ct += known.count_roots();
            retval.normalize(known);
            retval.bulk_ct -= retval.ids_used;
            return retval;
        }

        inline [[nodiscard]] partial_soln analyze(lin_alg::matrix_view const &input)
        {
            lin_alg::vector const line{ input };
            auto const len{ size(line) };
            partial_soln retval{
                    input,
                    std::vector<tree::node>(len)
                    //std::make_unique<id_key[]>(len),
                    //std::make_unique<id_key[]>(len)
            };
            {
                auto prev{ blank };
                auto data_iter{ begin(line) };
                for (auto &id : retval.left)
                {
                    if (!*data_iter++)
                        prev = blank;
                    else if (blank == prev)
                        prev = retval.ids_used++;
                    id = prev;
                }
            }
            retval.right = retval.left;
            return retval;
        }

        [[nodiscard]] partial_soln solve2(lin_alg::matrix_view const &data)
        {
            if (data.height().len > data.width().len)
                return solve2(transpose(data));
            if (1 == data.height().len)
                return analyze(data);
            else
            {
                assert(data.height().len);
                using lin_alg::all_t;
                auto const split_pt{ data.height().len / 2 };
                return
                    solve2(data[{all_t{}, { 0, split_pt }}]) |
                    solve2(data[{all_t{}, { static_cast<std::ptrdiff_t>(split_pt), data.height().len - split_pt }}]);
            }
        }
    }

    [[nodiscard]] std::size_t solve2(lin_alg::matrix_view const &data)
    {
        auto retval{ _internal::solve2(data) };
        return retval.bulk_ct + retval.ids_used;
    }

    namespace _internal
    {
        [[nodiscard]] auto emplace_zeros(lin_alg::matrix_view const &mat)
        {
            lin_alg::matrix_view retval;
            retval.height() = { 0, mat.height().len + 2, mat.height().stride };
            retval.width() = { 0, mat.width().len + 2, mat.width().stride };
            if (1 != retval.height().stride)
                retval.height().stride += 2;
            if (1 != retval.width().stride)
                retval.width().stride += 2;
            retval.backing = std::make_shared<lin_alg::data_t[]>(retval.height().len * retval.width().len);
            for (std::size_t index{ 0 }; index < mat.width().len; ++index)
                for (std::size_t jndex{ 0 }; jndex < mat.height().len; ++jndex)
                    retval[{index + 1, jndex + 1}].to_scalar() = mat[{index, jndex}].to_scalar();
            return retval;
        }

        void swarm_launch(lin_alg::matrix_view &data, std::size_t x, std::size_t y)
        {
            if (auto &cur_loc{ data[{x, y}].to_scalar() })
            {
                cur_loc = false;
                std::ptrdiff_t dx, dy;
                for (auto const coord : {&dx, &dy})
                    for (auto const dir : {-1, 1})
                    {
                        dx = dy = 0;
                        *coord = dir;
                        swarm_launch(data, x + dx, y + dy);
                    }
            }
        }
    }

    [[nodiscard]] std::size_t solve3(lin_alg::matrix_view const &data)
    {
        auto field{ _internal::emplace_zeros(data) };
        std::size_t count{ 0 };
        for (std::size_t index{ 0 }; index < field.width().len; ++index)
            for (std::size_t jndex{ 0 }; jndex < field.height().len; ++jndex)
                if (field[{index, jndex}])
                {
                    ++count;
                    _internal::swarm_launch(field, index, jndex);
                }
        return count;
    }
}