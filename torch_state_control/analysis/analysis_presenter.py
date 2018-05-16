import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Latex

from .analyst import Analyst


class AnalysisPresenter:

    def __init__(self, name, directory=None):
        self.analyst = Analyst(name, directory)

    # def __records__(self, checkpoint_id):
    #     if not checkpoint_id:
    #         checkpoint_id = self.__latest_record__().id
    #
    #     return self.tracer.backtrace_for(checkpoint_id)
    #
    # def __latest_record__(self):
    #     return self.tracer.latest()
    #
    # def losses(self, records):
    #     losses = []
    #
    #     for record in records:
    #         losses += record.losses_since_last_checkpoint
    #
    #     return losses

    # def plot_losses(self, checkpoint_id=None, show=True):
    #     records = self.__records__(checkpoint_id)
    #     losses = self.losses(records)
    #
    #     x_axis = list(range(len(losses)))
    #     y_axis = losses
    #
    #     # Plots.
    #     shared_plot_options = {
    #         'linestyle': 'solid',
    #         'marker': 'None',
    #         'markersize': 4
    #     }
    #     s = plt.plot(
    #         x_axis,
    #         y_axis,
    #         color='#00A6FB',
    #         label='Train set',
    #         **shared_plot_options
    #     )
    #
    #     # Fillings.
    #     # plt.fill_between(
    #     #     x_dev,
    #     #     0,
    #     #     y_dev,
    #     #     color='#00A6FB',
    #     #     alpha=1
    #     # )
    #
    #     # s.yaxis.tick_right()
    #
    #     # Labels.
    #     plt.xlabel('Epochs')
    #     plt.ylabel("Loss")
    #
    #     # plt.ylim(ymax=0.0003, ymin=0)
    #     plt.title(f"Loss development over {len(losses)} epochs")
    #     plt.legend(loc='upper right')
    #
    #     if show:
    #         plt.show()
    #
    # def plot_current_performance(self, checkpoint_id=None, show=True):
    #     records = self.__records__(checkpoint_id)
    #     record = records[-1]
    #
    #     # print(f"Performance on Train set: -- {record.train_set_performance:f}")
    #     # print(f"Performance on Dev set: ---- {record.dev_set_performance:f}")
    #
    #     lengths = [record.train_set_performance, record.dev_set_performance]
    #     # plt.suptitle('Length classes')
    #     # plt.xlabel('Length')
    #     # plt.ylabel('Amount of records')
    #
    #     # plt.hist(
    #     #     lengths,
    #     #     color='green',
    #     #     align='mid'
    #     # )
    #
    #
    #     # fig, axes = plt.subplots(nrows=2, ncols=2)
    #     # ax0, ax1, ax2, ax3 = axes.flatten()
    #
    #     colors = ['red', 'lime']
    #     plt.hist([[0, 1], [1, 1.1]], density=False, histtype='bar')
    #     # ax0.legend(prop={'size': 10})
    #     # ax0.set_title('bars with legend')
    #
    #     if show:
    #         plt.show()

    def plot_performances(self, checkpoint=None, show=True):
        train_set_performances, dev_set_performances = self.analyst.performances(checkpoint)
        losses = self.analyst.losses(checkpoint)

        x_train = list(range(len(train_set_performances)))
        y_train = train_set_performances
        x_dev = list(range(len(dev_set_performances)))
        y_dev = dev_set_performances

        # Plots.
        shared_plot_options = {
            'linestyle': 'None',
            'marker': '|',
            'markersize': 4
        }
        plt.plot(
            x_train,
            y_train,
            color='#04151F',
            label=f"Train set, {y_train[-1]:f}",
            **shared_plot_options
        )
        plt.plot(
            x_dev,
            y_dev,
            color='#00A6FB',
            label=f"Dev set, {y_dev[-1]:f}",
            **shared_plot_options
        )

        # plt.annotate(
        #     'THE DAY I REALIZED\nI COULD COOK BACON\nWHENEVER I WANTED',
        #     xy=(x_train[-1], y_train[-1]),
        #     arrowprops=dict(arrowstyle='->'),
        #     # xytext=(15, -10)
        # )

        # Fillings.
        plt.fill_between(
            x_dev,
            0,
            y_dev,
            color='#00A6FB',
            alpha=1
        )
        plt.fill_between(
            x_train,
            0,
            y_train,
            color='#04151F',
            alpha=0.85
        )

        # Labels.
        plt.xlabel('Checkpoints')
        plt.ylabel('Loss')

        plt.title(f"Performance on Train- and Dev set over {len(train_set_performances)} checkpoints and {len(losses)} epochs")
        plt.legend(loc='upper right')

        if show:
            plt.show()

    def plot_changelog(self, checkpoint=None):
        changelog = self.analyst.changelog(checkpoint)

        display(Markdown("### Changelog"))

        for entry in changelog:
            display(Markdown(f"#### Checkpoint #{entry['checkpoint_id']}"))
            display(Markdown(entry['notes']))

    def plot_analysis(self, checkpoint=None):
        plt.figure(num=0, figsize=(10, 10))

        # cols = 3
        # rows = math.ceil(len(indices) / cols)
        #
        # plt.figure(num=0, figsize=(10, 1.5 * rows))

        grid = (3, 1)

        # sp = plt.subplot(*grid, 1)
        # self.plot_current_performance(checkpoint, show=False)
        #
        # sp = plt.subplot(*grid, 2)
        # self.plot_losses(checkpoint, show=False)
        # sp.set_aspect('equal')
        sp = plt.subplot(*grid, 3)
        self.plot_performances(checkpoint, show=False)



        # sp = plt.subplot(*grid, 3)
        # self.plot_current_performance(checkpoint, show=False)


        # plt.axis('on')
        # sp.set_xticklabels([])
        # sp.set_yticklabels([])
        # sp.set_aspect('equal')
        plt.tight_layout()
        plt.show()

        self.plot_changelog(checkpoint)
