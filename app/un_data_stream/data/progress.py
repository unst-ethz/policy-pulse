import sys
import time


def progressbar(it, total=None, prefix="", size=60, out=sys.stdout):
    """Wrap an iterator loop with a printing progress bar. Courtesy of:
    https://stackoverflow.com/questions/3160699/python-progress-bar

    Parameters
    ----------
    it : Iterator
        Iterator to loop over. If this is a generator, the total must be provided.
    total : int, optional
        Total number of items in the iterator, required to calculate progress percentage
    prefix : str, optional
        A prefix to be prepended before the progress bar, by default ""
    size : int, optional
        The width of the progress bar, by default 60
    out : _type_, optional
        The output stream to write the progress bar to, by default sys.stdout

    Yields
    ------
    _type_
        The items from the input iterator, with the progress bar printed to the output stream
    """
    count = total if total is not None else len(it)
    if count == 0:
        print(f"{prefix}[{'.' * size}] 0/0 Est wait 00:00.0", file=out, flush=True)
        return

    # Avoid stdout becoming a bottleneck for very large iterables by only
    # updating a maximum of 5 times for each block.
    update_every = max(1, count // (size * 5))
    start = time.time()  # time estimate start

    def show(j):
        x = int(size * j / count)
        # time estimate calculation and string
        remaining = ((time.time() - start) / j) * (count - j)
        mins, sec = divmod(remaining, 60)  # limited to minutes
        time_str = f"{int(mins):02}:{sec:03.1f}"
        print(
            f"{prefix}[{'█' * x}{('.' * (size - x))}] {j}/{count} Est wait {time_str}",
            end="\r",
            file=out,
            flush=True,
        )

    show(0.1)  # avoid div/0
    for i, item in enumerate(it):
        yield item
        new_i = i + 1
        if (new_i % update_every == 0) or (new_i == count):
            show(new_i)
    print("\n", flush=True, file=out)
