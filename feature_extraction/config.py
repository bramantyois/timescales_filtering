import seaborn as sns


def config_plotting(context="paper", palette="Set3"):
    sns.set_style("darkgrid")
    if context == "paper":
        sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    elif context == "talk":
        sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})
    else:
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    sns.set_palette(palette)
