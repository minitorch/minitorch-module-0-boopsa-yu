"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable
from collections.abc import Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """浮点数乘法。

    Args:
        x: 第一个参数
        y: 第二个参数

    Returns:
        返回 x * y

    """
    return x * y


def id(x: float) -> float:
    """恒等函数

    Args:
        x: 输入参数

    Returns:
        返回参数本身

    """
    return x


def add(x: float, y: float) -> float:
    """两数之和。

    Args:
        x: 第一个参数
        y: 第二个参数

    Returns:
        返回 x + y

    """
    return x + y


def neg(x: float) -> float:
    """Negates a number

    Args:
        x: 输入参数

    Returns:
        输入参数的相反数

    """
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another

    Args:
        x: one number
        y: another number

    Returns:
        x < y

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal

    Args:
        x: one number
        y: another number

    Returns:
        x == y

    """
    return x == y


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers

    Args:
        x: one number
        y: another number

    Returns:
        max(x, y)

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value

    Args:
        x: one number
        y: another number

    Returns:
        两个数字的数值差距是否在 0.01 以内

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    r"""计算数值稳定的 Sigmoid 激活函数。

    为了防止在计算指数函数 $e^x$ 时发生浮点数溢出（Overflow）,
    本实现对正负输入进行了分段处理：
    - 当 $x \ge 0$ 时，使用 $\frac{1}{1 + e^{-x}}$ 避免 $e^x$ 溢出。
    - 当 $x < 0$ 时，使用 $\frac{e^x}{1 + e^x}$ 避免 $e^{-x}$ 溢出。

    Args:
        x: 输入的实数或激活值。

    Returns:
        映射到 $(0, 1)$ 区间内的浮点数结果。

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    r"""应用 ReLU（修正线性单元）激活函数。

    数学表达式为：
    $$f(x) = \max(0, x)$$

    当输入 $x > 0$ 时保持不变，当 $x \le 0$ 时截断为 0。

    Args:
        x: 输入的单个浮点数激活值。

    Returns:
        经过 ReLU 激活后的结果。

    """
    return max(0.0, x)


def log(x: float) -> float:
    r"""计算 x 的自然对数

    Args:
        x: 输入的实数

    Returns:
        返回 x 的自然对数 $\log(x)$

    """
    return math.log(x)


def exp(x: float) -> float:
    r"""计算 x 的自然指数

    Args:
        x: 输入的实数

    Returns:
        返回 x 的自然对数 $e^x$

    """
    return math.exp(x)


def inv(x: float) -> float:
    r"""计算 x 倒数

    Args:
        x: 输入的实数

    Returns:
        返回 $frac{1}{x}$

    """
    return 1.0 / x


def log_back(x: float, grad_out: float) -> float:
    r"""计算自然对数算子的反向传播梯度。

    Args:
        x: 前向传播时的输入值（必须大于 0）。
        grad_out: 从下一层传回的上游梯度（a second arg）。

    Returns:
        传回给输入 x 的下游梯度。

    """
    return (1.0 / x) * grad_out


def inv_back(x: float, grad_out: float) -> float:
    r"""计算倒数的反向传播梯度。

    Args:
        x: 前向传播时的输入值（不能为 0）。
        grad_out: 从下一层传回的上游梯度（a second arg）。

    Returns:
        传回给输入 x 的下游梯度。

    """
    return (-1.0 / x**2) * grad_out


def relu_back(x: float, grad_out: float) -> float:
    r"""计算 ReLU 激活函数的反向传播梯度。

    Args:
        x: 前向传播时的输入值。
        grad_out: 从下一层传回的上游梯度（a second arg）。

    Returns:
        传回给输入 x 的下游梯度。

    """
    # 如果前向传播的输入大于 0，梯度原样传回；否则梯度直接截断为 0
    return grad_out if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable, xs: Iterable) -> list:
    """Applies a given function to each element of an iterable。

    Args:
        fn: 提供的函数
        xs: 可迭代的目标对象

    Returns:
        返回将 fn 作用在每个 xs 对象后的列表

    """
    return [fn(x) for x in xs]


def zipWith(fn: Callable, xs: Iterable, ys: Iterable) -> list:
    """Combines elements from two iterables using a given function

    Args:
        fn: 给定函数
        xs: 第一个可迭代对象
        ys: 第二个可迭代对象

    Returns:
        返回将 fn 作用在每个 xs 和 ys 对象后的列表

    """
    return [fn(x, y) for x, y in zip(xs, ys)]


def reduce(fn: Callable, xs: Iterable, start: float) -> float:
    """规约函数，逐渐把整个可迭代对象通过指定的函数合并成一个

    Args:
        fn: 指定的函数
        xs: 目标迭代对象
        start: 规约的初始值

    Returns:
        The final value obtained by repeatedly applying fn to elements
        in xs.

    """
    result = start
    for x in xs:
        result = fn(result, x)

    return result


def negList(xs: list) -> list:
    """将列表的所有元素取反

    Args:
        xs: 目标列表

    Returns:
        所有元素取反后的列表

    """
    return map(neg, xs)


def addLists(xs: list, ys: list) -> list:
    """逐元素加法

    Args:
        xs: 目标列表
        ys: 目标列表

    Returns:
        返回逐元素相加后的列表

    """
    return zipWith(add, xs, ys)


def sum(xs: list) -> float:
    """对列表中的元素求和

    Args:
        xs: 目标列表

    Returns:
        求和后的结果 sum(xs)

    """
    return reduce(add, xs, 0.0)


def prod(xs: list) -> float:
    """求列表中的元素的乘积

    Args:
        xs: 目标列表

    Returns:
        所有元素乘积后的结果 prod(xs)

    """
    return reduce(mul, xs, 1.0)
