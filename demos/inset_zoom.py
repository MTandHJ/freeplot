
# inset zoom demo

from freeplot.base import FreePlot


fp = FreePlot((1, 1))

fp.lineplot([1, 2, 3], [4, 5, 6], label='a')
fp.lineplot([1, 2, 3], [3, 5, 7], label='b')
axins, patch, lines = fp.inset_axes(
    xlims=(1.9, 2.1),
    ylims=(4.9, 5.1),
    bounds=(0.1, 0.7, 0.2, 0.2),
    index=(0, 0),
    style='line'
)
fp.lineplot([1, 2, 3], [4, 5, 6], index=axins)
fp.lineplot([1, 2, 3], [3, 5, 7], index=axins)
# for line in lines:
#     line.set({'linewidth':5, 'alpha': 0.7})
fp.show()