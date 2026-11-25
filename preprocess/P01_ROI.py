"""
USER DEFINED ROI AND REMOVE ALL DETECTIONS OUTSIDE
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
from matplotlib.patches import Circle


def select_circular_roi(data, max_frame=3000, unit=None):
    """
    Select a circular ROI interactively.

    1. Click to define the center.
    2. Move the mouse to preview the radius.
    3. Click to define the radius.
    4. Press Enter to validate, or Escape to cancel.

    Returns
    -------
    (cx, cy, r) : tuple of floats
        Center coordinates and radius, in the same units as data["x"]
        and data["y"], potentially explicit in arg units. Returns None if cancelled.
    """
    plot_data = data[data["frame"] < max_frame]

    fig, ax = plt.subplots()

    ax.plot(plot_data["x"], plot_data["y"], ".", markersize=2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title( "Click the center, then the edge. "
        "Press Enter to validate, Escape to cancel.")
    if unit is not None:
        ax.set_xlabel(f"x ({unit})")
        ax.set_ylabel(f"y ({unit})")

    state = {
        "center": None, "radius": None, "result": None,
    }

    circle = Circle((0, 0), 0, fill=False, color="red", linewidth=2,visible=False)
    ax.add_patch(circle)

    def update_circle(x, y):
        cx, cy = state["center"]
        r = np.hypot(x - cx, y - cy)

        circle.set_center((cx, cy))
        circle.set_radius(r)
        circle.set_visible(True)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return

        if state["center"] is None or state["radius"] is not None:
            # First click: define the center
            state["center"] = (event.xdata, event.ydata)
            state["radius"] = None #restart radius
            circle.set_center(state["center"])
            circle.set_visible(True)
            fig.canvas.draw_idle()

        else:
            # Second click: define the radius
            cx, cy = state["center"]
            r = np.hypot(event.xdata - cx,event.ydata - cy)

            if r > 0:
                state["radius"] = r
                update_circle(event.xdata, event.ydata)
                ax.set_title( "Circle ready. Press Enter to validate, "
                    "or click again to select new center.")

    def on_motion(event):
        if (
            state["center"] is not None
            and state["radius"] is None 
            and event.inaxes == ax
            and event.xdata is not None
        ):
            update_circle(event.xdata, event.ydata)

    def on_key(event):
        if event.key == "enter":
            if state["center"] and state["radius"]:
                cx, cy = state["center"]
                state["result"] = (cx, cy, state["radius"])
                plt.close(fig)

        elif event.key == "escape":
            state["result"] = None
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    return state["result"]



def filter_remove_outside_roi(df, cx, cy, r):
    """
    Remove detections outside the circular ROI.
    """
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    dist_sq = (x - cx)**2 + (y - cy)**2
    r_sq = r**2

    mask = dist_sq <= r_sq

    return df.loc[mask].reset_index(drop=True)
