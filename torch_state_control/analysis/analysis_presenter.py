import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Latex

from .analyst import Analyst


class AnalysisPresenter:

    PLOT_SIZE = 4

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

    def plot_loss(self, checkpoint=None, show=True):
        if show:
            grid = (1, 1)
            fig_width = 2 * self.PLOT_SIZE
            fig_height = grid[0] * self.PLOT_SIZE
            plt.figure(num=0, figsize=(fig_width, fig_height))

            plt.subplot(*grid, 1)

        losses = self.analyst.losses(checkpoint)

        x_axis = list(range(len(losses)))
        y_axis = losses

        conditional_plot_options = {
            'marker': '|'
        }
        if len(losses) > 150:
            conditional_plot_options['marker'] = 'None'

        plt.plot(
            x_axis,
            y_axis,
            color='#04151F',
            # label='Train set',
            linestyle='solid',
            markersize=4,
            **conditional_plot_options
        )

        # Labels.
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        # plt.ylim(ymax=0.0003, ymin=0)
        plt.title(f"Loss development over {len(losses)} epochs")
        # plt.legend(loc='upper right')

        if show:
            plt.tight_layout()
            plt.show()

    def plot_confusions(self, checkpoint=None, show=True):
        if show:
            grid = (2, 1)
            fig_width = 2 * self.PLOT_SIZE
            fig_height = grid[0] * self.PLOT_SIZE
            plt.figure(num=0, figsize=(fig_width, fig_height))

            plt.subplot(*grid, 1)

        train_set_confusions, dev_set_confusions = self.analyst.confusions(checkpoint)

        amount_of_recognized_frauds_train = [1 - tp / (tp + fn) for tp, fp, tn, fn in train_set_confusions]
        amount_of_recognized_frauds_dev = [1 - tp / (tp + fn) for tp, fp, tn, fn in dev_set_confusions]

        # Plots.
        shared_plot_options = {
            'linestyle': 'None',
            'marker': '|',
            'markersize': 4
        }
        plt.plot(
            amount_of_recognized_frauds_dev,
            color='#52EF2B',
            label=f"Dev set - {amount_of_recognized_frauds_dev[-1]*100:.2f}%",
            **shared_plot_options
        )
        plt.plot(
            amount_of_recognized_frauds_train,
            color='#0AD6FF',
            label=f"Train set - {amount_of_recognized_frauds_train[-1]*100:.2f}%",
            **shared_plot_options
        )

        # Fillings.
        plt.fill_between(
            list(range(len(amount_of_recognized_frauds_dev))),
            0,
            amount_of_recognized_frauds_dev,
            color='#52EF2B',
            alpha=1
        )
        plt.fill_between(
            list(range(len(amount_of_recognized_frauds_train))),
            0,
            amount_of_recognized_frauds_train,
            color='#0AD6FF',
            alpha=0.95
        )

        # Labels.
        plt.xlabel("Checkpoints")
        plt.ylabel("Amount in %")
        plt.title(f"Unidentified Positives over {len(train_set_confusions)} checkpoints")
        plt.legend(loc='best')

        if show:
            plt.subplot(*grid, 2)

        amount_of_falsley_accused_train = [fp / (fp + tn) for tp, fp, tn, fn in train_set_confusions]
        amount_of_falsley_accused_dev = [fp / (fp + tn) for tp, fp, tn, fn in dev_set_confusions]

        # Plots.
        shared_plot_options = {
            'linestyle': 'None',
            'marker': '|',
            'markersize': 4
        }
        plt.plot(
            amount_of_falsley_accused_dev,
            color='#F42A13',
            label=f"Dev set - {amount_of_falsley_accused_dev[-1]*100:.2f}%",
            **shared_plot_options
        )
        plt.plot(
            amount_of_falsley_accused_train,
            color='#211E1D',
            label=f"Train set - {amount_of_falsley_accused_train[-1]*100:.2f}%",
            **shared_plot_options
        )

        # Fillings.
        plt.fill_between(
            list(range(len(amount_of_falsley_accused_dev))),
            0,
            amount_of_falsley_accused_dev,
            color='#F42A13',
            alpha=1
        )
        plt.fill_between(
            list(range(len(amount_of_falsley_accused_train))),
            0,
            amount_of_falsley_accused_train,
            color='#211E1D',
            alpha=0.95
        )

        # Labels.
        plt.xlabel("Checkpoints")
        plt.ylabel("Amount in %")
        plt.title(f"Falsely accused over {len(train_set_confusions)} checkpoints")
        plt.legend(loc='best')

        if show:
            plt.tight_layout()
            plt.show()

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
        if show:
            grid = (1, 1)
            fig_width = 2 * self.PLOT_SIZE
            fig_height = grid[0] * self.PLOT_SIZE
            plt.figure(num=0, figsize=(fig_width, fig_height))

            plt.subplot(*grid, 1)

        train_set_performances, dev_set_performances = self.analyst.performances(checkpoint)
        # Expect the performances to be single floats for this graph.
        train_set_performances = [float(perf) for perf in train_set_performances]
        dev_set_performances = [float(perf) for perf in dev_set_performances]

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
            plt.tight_layout()
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
        # self.plot_loss(checkpoint, show=False)



        # sp = plt.subplot(*grid, 3)
        # self.plot_current_performance(checkpoint, show=False)


        # plt.axis('on')
        # sp.set_xticklabels([])
        # sp.set_yticklabels([])
        # sp.set_aspect('equal')
        plt.tight_layout()
        plt.show()

        self.plot_changelog(checkpoint)
