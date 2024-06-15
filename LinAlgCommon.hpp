#pragma once
#include <bitset>
#include <utility>

#include "Tweaks.hpp"

namespace lin_alg
{
    struct all_t {};
    namespace _internal
    {
        template<typename inheritor, typename slice, std::size_t dim>
        struct overload_insertion abstract final
        {
            template<std::uintmax_t code>
            struct _overloads abstract
            {
                template<typename ISeq> struct unpack_code;
                template<std::size_t... nums>
                struct unpack_code<std::index_sequence<nums...>>
                {
                private:
                    constexpr static std::bitset<sizeof...(nums)> bs{ code };
                public:
                    typedef std::tuple<std::conditional_t<bs[nums], all_t, slice>...> type;
                };
                typedef unpack_code<std::make_index_sequence<dim>>::type arg_t;
            private:
                CRTP_ACCESS(inheritor);
            public:
                constexpr [[nodiscard]] auto operator[](arg_t const &masks)
                {
                    return _virtual()->op_bracket(masks, std::make_index_sequence<dim>{});
                }
                constexpr [[nodiscard]] auto const operator[](arg_t const &masks) const
                {
                    return const_cast<_overloads &>(*this)[masks];
                }
            };

            template<std::uintmax_t code = (1 << dim) - 1>
            struct recurse abstract : public recurse<code - 1>, public _overloads<code>
            {
                using _overloads<code>::operator[], recurse<code - 1>::operator[];
            };
            template<> struct recurse<0> abstract : public _overloads<0> {};
        };

        template<typename inheritor, typename slice, std::size_t dim>
        using _insert_overloads = overload_insertion<inheritor, slice, dim>::template recurse<>;

        template<typename inheritor, std::size_t> class tensor_base {};
        template<typename inheritor>
        class tensor_base<inheritor, 2>
        {
            CRTP_ACCESS(inheritor);
            enum { HORIZ = 0, VERT = 1 };
        public:
            [[nodiscard]] auto &width() noexcept { return _virtual()->indices[HORIZ]; }
            [[nodiscard]] auto const &width() const noexcept
            {
                return const_cast<tensor_base *>(this)->width();
            }
            [[nodiscard]] auto &height() noexcept { return _virtual()->indices[VERT]; }
            [[nodiscard]] auto const &height() const noexcept
            {
                return const_cast<tensor_base *>(this)->height();
            }
            [[nodiscard]] auto rows_begin() noexcept { return _virtual()->begin<HORIZ>(); }
            [[nodiscard]] auto rows_begin() const noexcept { return _virtual()->cbegin<HORIZ>(); }
            [[nodiscard]] auto rows_cbegin() const noexcept { return rows_begin(); }
            [[nodiscard]] auto rows_end() noexcept { return _virtual()->end<HORIZ>(); }
            [[nodiscard]] auto rows_end() const noexcept { return _virtual()->cend<HORIZ>(); }
            [[nodiscard]] auto rows_cend() noexcept { return rows_end(); }
            [[nodiscard]] auto cols_begin() noexcept { return _virtual()->begin<HORIZ>(); }
            [[nodiscard]] auto cols_begin() const noexcept { return _virtual()->cbegin<HORIZ>(); }
            [[nodiscard]] auto cols_cbegin() const noexcept { return cols_begin(); }
            [[nodiscard]] auto cols_end() noexcept { return _virtual()->end<HORIZ>(); }
            [[nodiscard]] auto cols_end() const noexcept { return _virtual()->cend<HORIZ>(); }
            [[nodiscard]] auto cols_cend() noexcept { return cols_end(); }
        };
    }
}
