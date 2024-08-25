import logging
import warnings
from functools import partial

import pyqtgraph.dockarea
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from PyQt5 import QtWidgets as qtw
from pyqtgraph.dockarea import DockArea
from stylesheets import (
    DOCK_LABEL_DIM_STYLESHEET,
    DOCK_LABEL_STYLESHEET,
    DOCK_STATUS_STYLESHEET,
    DOCK_TITLE_STYLESHEET,
)

logger = logging.getLogger("burst-detector")


def _widget_position(widget):  # pragma: no cover
    return widget.parentWidget().mapToGlobal(widget.geometry().topLeft())


# redefining pyqtgraph class so color can be changed
class DropAreaOverlay(qtw.QWidget):
    """Overlay widget that draws drop areas during a drag-drop operation"""

    def __init__(self, parent):
        qtw.QWidget.__init__(self, parent)
        self.dropArea = None
        self.hide()
        self.setAttribute(qtc.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def setDropArea(self, area):
        self.dropArea = area
        if area is None:
            self.hide()
        else:
            ## Resize overlay to just the region where drop area should be displayed.
            ## This works around a Qt bug--can't display transparent widgets over QGLWidget
            prgn = self.parent().rect()
            rgn = qtc.QRect(prgn)
            w = min(30, int(prgn.width() / 3))
            h = min(30, int(prgn.height() / 3))

            if self.dropArea == "left":
                rgn.setWidth(w)
            elif self.dropArea == "right":
                rgn.setLeft(rgn.left() + prgn.width() - w)
            elif self.dropArea == "top":
                rgn.setHeight(h)
            elif self.dropArea == "bottom":
                rgn.setTop(rgn.top() + prgn.height() - h)
            elif self.dropArea == "center":
                rgn.adjust(w, h, -w, -h)
            self.setGeometry(rgn)
            self.show()

        self.update()

    def paintEvent(self, ev):
        if self.dropArea is None:
            return
        p = qtg.QPainter(self)
        rgn = self.rect()

        p.setBrush(qtg.QBrush(qtg.QColor(0, 120, 215, 77)))
        p.setPen(qtg.QPen(qtg.QColor(0, 120, 215, 77), 2))
        p.drawRect(rgn)
        p.end()


# redefining pyqtgraph class to change style and label height
class DockLabel(pyqtgraph.dockarea.DockLabel):
    def __init__(
        self,
        text,
        closable=False,
        fontSize="12px",
        ss=DOCK_LABEL_STYLESHEET,
        dim_ss=DOCK_LABEL_DIM_STYLESHEET,
    ):
        self.ss = ss
        self.dim_ss = dim_ss
        super().__init__(text, closable, fontSize)

    def updateStyle(self):
        self.setStyleSheet(self.ss) if not self.dim else self.setStyleSheet(self.dim_ss)

    def paintEvent(self, ev):
        p = qtg.QPainter(self)
        rgn = self.contentsRect()
        align = self.alignment()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.hint = p.drawText(rgn, align, self.text())
        p.end()

        self.setFixedHeight(self.hint.height() + 10)


class Dock(pyqtgraph.dockarea.Dock):
    confirm_before_close_view = True

    def __init__(
        self,
        name,
        area=None,
        widget=None,
        hideTitle=False,
        autoOrientation=False,
        label=None,
        closable=False,
        **kargs
    ):
        super().__init__(
            name,
            area,
            widget=widget,
            hideTitle=hideTitle,
            autoOrientation=autoOrientation,
            label=label,
            **kargs
        )
        self._widget = widget
        self._dock_widgets = {}
        self.closable = closable
        self.label = label

    def add_button(
        self,
        callback=None,
        text=None,
        icon=None,
        checkable=False,
        checked=False,
        event=None,
        name=None,
    ):

        name = name or getattr(callback, "__name__", None) or text
        assert name
        button = qtw.QPushButton(chr(int(icon, 16)) if icon else text)
        button.setCheckable(checkable)
        if checkable:
            button.setChecked(checked)
        button.setToolTip(name)

        button.clicked.connect(callback)

        assert name not in self._dock_widgets
        self._dock_widgets[name] = button
        self._buttons_layout.addWidget(button, 1)

        return button

    def _create_menu(self):
        self._menu = qtw.QMenu("%s menu" % self.objectName(), self)

    def _create_buttons(self):
        self._buttons = qtw.QWidget(self)
        self._buttons_layout = qtw.QHBoxLayout(self._buttons)
        self._buttons_layout.setDirection(1)
        self._buttons_layout.setContentsMargins(0, 0, 0, 0)
        self._buttons_layout.setSpacing(0)
        self._buttons.setLayout(self._buttons_layout)

        self._default_buttons()

    def _create_title_bar(self):
        self._title_bar = qtw.QWidget(self)
        self._title_bar.setStyleSheet(DOCK_TITLE_STYLESHEET)

        self._layout = qtw.QHBoxLayout(self._title_bar)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # label
        self._layout.addWidget(self.label, 100)
        self._layout.addStretch(1)

        # buttons
        self._buttons = qtw.QWidget()
        self._buttons_layout = qtw.QHBoxLayout(self._buttons)
        self._buttons_layout.setDirection(1)
        self._buttons_layout.setContentsMargins(0, 0, 0, 0)
        self._buttons_layout.setSpacing(0)
        self._buttons.setLayout(self._buttons_layout)

        self._default_buttons()

        self._layout.addWidget(self._buttons, 0)
        self._title_bar.setLayout(self.layout)

    def _default_buttons(self):
        if self.closable:
            self.add_button(callback=self.close, name="close", text="✕")

        def on_view_menu():
            button = self._dock_widgets["view_menu"]
            x = _widget_position(button).x()
            y = _widget_position(self._widget).y()
            self._menu.exec(qtc.QPoint(x, y))

        self.add_button(callback=on_view_menu, name="view_menu", text="Menu")

    def get_widget(self, name):
        return self._dock_widgets[name]

    @property
    def status(self):
        return self._status.text()

    def set_status(self, text):
        self._status.setText(text)

    def updateStyle(self):
        # redraws titlebar after widget has moved
        if self.labelHidden:
            self.widgetArea.setStyleSheet(self.nStyle)
        else:
            if self.moveLabel:
                self._layout.insertWidget(0, self.label, 100)
                self.topLayout.insertWidget(0, self._title_bar)
            self.widgetArea.setStyleSheet(self.hStyle)
        pass


def _create_dock(name, widget, closable=False):

    label = DockLabel(
        name, closable=False
    )  # closable always false, we make our own close button
    dock = Dock(
        name, widget=widget, label=label, autoOrientation=False, closable=closable
    )

    dock._create_menu()
    dock._create_title_bar()

    # create new layout
    widget_container = qtw.QWidget(dock)
    widget_layout = qtw.QVBoxLayout(widget_container)
    widget_layout.setContentsMargins(0, 0, 0, 0)
    widget_layout.setSpacing(0)

    widget_layout.addWidget(dock._title_bar, 1)
    widget_layout.addWidget(dock.widgetArea, 100)
    # status bar
    dock._status = qtw.QLabel("")
    dock._status.setMaximumHeight(30)
    dock._status.setStyleSheet(DOCK_STATUS_STYLESHEET)
    widget_layout.addWidget(dock._status, 1)

    # clear old layout
    dock.topLayout.removeWidget(dock.widgetArea)
    dock.topLayout.removeWidget(label)
    qtw.QWidget().setLayout(dock.topLayout)

    # set new layout
    dock.topLayout = widget_layout
    dock.setLayout(dock.topLayout)

    dock.dockdrop.overlay = DropAreaOverlay(dock.dockdrop.dndWidget)
    dock.dockdrop.overlay.raise_()

    widget.setDock(dock)

    return dock


class MainWindow(qtw.QMainWindow):
    "Main application window"

    def __init__(
        self,
        controller,
        position=None,
        size=None,
        name=None,
        subtitle=None,
        enable_threading=True,
    ):

        self.controller = controller
        self._closed = False
        self._enable_threading = enable_threading
        if not qtw.QApplication.instance():
            raise RuntimeError("A Qt application must be created.")
        super().__init__()

        self._set_name(name, str(subtitle or ""))
        position = position or (200, 200)
        size = size or (800, 600)
        self._set_pos_size(position, size)

        # status bar
        self._lock_status = False
        self._status_bar = qtw.QStatusBar(self)
        self.setStatusBar(self._status_bar)

        self._create_menu()

        self.dockArea = DockArea(parent=self)
        self.setCentralWidget(self.dockArea)

        self.show()

    def a(self, s):
        logger.info(s)

    def _create_menu(self):
        menu_bar = self.menuBar()

        # File
        self.file_menu = menu_bar.addMenu("File")
        self.file_menu.addAction(
            "Load Recording",
            self.controller.load_recording,
            qtg.QKeySequence(qtc.Qt.CTRL + qtc.Qt.Key_O),
        )

        # Edit
        self.edit_menu = menu_bar.addMenu("Edit")
        self.edit_menu.addAction(
            "Merge", self.controller.merge, qtg.QKeySequence(qtc.Qt.Key_G)
        )
        self.edit_menu.addSeparator()

        self.add_cur_menu("Move selected", "good", "selected", qtc.Qt.Key_G)
        self.add_cur_menu("Move selected", "mua", "selected", qtc.Qt.Key_M)
        self.add_cur_menu("Move selected", "noise", "selected", qtc.Qt.Key_N)

        self.edit_menu.addAction(
            "Move selected to unsorted",
            partial(self.controller.label_clusters, grp="selected", label="unsorted"),
            qtg.QKeySequence(qtc.Qt.CTRL + qtc.Qt.Key_U),
        )

        self.edit_menu.addSeparator()
        self.edit_menu.addAction(
            "Move all to good",
            partial(self.controller.label_clusters, "all", label="good"),
            qtg.QKeySequence(qtc.Qt.CTRL + qtc.Qt.ALT + qtc.Qt.Key_G),
        )
        self.edit_menu.addAction(
            "Move all to mua",
            partial(self.controller.label_clusters, "all", label="mua"),
            qtg.QKeySequence(qtc.Qt.CTRL + qtc.Qt.ALT + qtc.Qt.Key_M),
        )
        self.edit_menu.addAction(
            "Move all to noise",
            partial(self.controller.label_clusters, "all", label="noise"),
            qtg.QKeySequence(qtc.Qt.CTRL + qtc.Qt.ALT + qtc.Qt.Key_N),
        )
        self.edit_menu.addAction(
            "Move all to unsorted",
            partial(self.controller.label_clusters, "all", label="unsorted"),
            qtg.QKeySequence(qtc.Qt.CTRL + qtc.Qt.ALT + qtc.Qt.Key_U),
        )
        self.edit_menu.addSeparator()
        self.edit_menu.addAction("Filter with condition file")

        # Extensions
        self.ext_menu = menu_bar.addMenu("Extensions")
        self.ext_menu.addAction("Kilosort", self.kilo_demo)
        self.ext_menu.addSeparator()
        self.ext_menu.addAction("Add extension...")

    def kilo_demo(self):
        dialog = qtw.QDialog()
        dialog.setWindowTitle("Set Kilosort parameters")
        QBtn = qtw.QDialogButtonBox.Ok | qtw.QDialogButtonBox.Cancel

        buttonBox = qtw.QDialogButtonBox(QBtn)

        layout = qtw.QGridLayout()
        layout.addWidget(buttonBox, 10, 3)
        layout.addWidget(qtw.QPushButton("Advanced >>"), 10, 0)
        label = qtw.QLabel("Probe layout")
        label.setToolTip("Path to probe channel map")
        layout.addWidget(label, 0, 0)
        layout.addWidget(qtw.QLineEdit(""), 0, 1)

        layout.addWidget(qtw.QLabel("Time range"), 0, 2)
        layout.addWidget(qtw.QLineEdit(""), 0, 3)

        layout.addWidget(qtw.QLabel("N blocks for registration"), 1, 0)
        layout.addWidget(qtw.QLineEdit(""), 1, 1)

        layout.addWidget(qtw.QLabel("Threshold"), 1, 2)
        layout.addWidget(qtw.QLineEdit(""), 1, 3)

        layout.addWidget(qtw.QLabel("Lambda"), 2, 0)
        layout.addWidget(qtw.QLineEdit(""), 2, 1)

        layout.addWidget(qtw.QLabel("AUC for splits"), 2, 2)
        layout.addWidget(qtw.QLineEdit(""), 2, 3)

        dialog.setLayout(layout)
        dialog.exec()

    def add_cur_menu(self, stem, label, grp, key):
        menu = self.edit_menu.addMenu(stem + " to " + label)
        menu.addAction(
            "No curation label",
            partial(self.controller.label_clusters, grp=grp, label=label),
            qtg.QKeySequence(qtc.Qt.CTRL + key),
        )
        menu.addSeparator()
        add_sep = menu.addSeparator()
        menu.addAction(
            "Add curation label...",
            partial(
                self.add_cur_label, menu=menu, grp=grp, label=label, add_sep=add_sep
            ),
        )

    def add_cur_label(self, menu, grp, label, add_sep):
        cur_label, ok = qtw.QInputDialog.getText(
            self,
            "Add curation label",
            "Add a new curation label for %s clusters" % label,
        )
        if ok:
            cur_act = qtw.QAction(cur_label, parent=menu)
            cur_act.triggered.connect(
                partial(
                    self.controller.label_clusters,
                    grp=grp,
                    label=label,
                    cur_label=cur_label,
                )
            )

            # shortcut
            shortcut, ok = get_key_sequence(cur_label)
            logger.info(ok)
            if ok:
                cur_act.setShortcut(shortcut)

            menu.insertAction(add_sep, cur_act)

    def _set_name(self, name, subtitle):
        if name is None:
            name = self.__class__.__name__
        title = name if not subtitle else name + " - " + subtitle
        self.setWindowTitle(title)
        self.setObjectName(name)

        self.name = name

    def _set_pos_size(self, position, size):
        if position is not None:
            self.move(position[0], position[1])
        if size is not None:
            self.resize(qtc.QSize(size[0], size[1]))


# TODO: keep track of all used shortcuts, make sure no duplicate shortcuts are created
def get_key_sequence(action):
    dialog = qtw.QDialog()
    dialog.setWindowTitle("Create shortcut")
    form = qtw.QFormLayout(dialog)

    form.addRow(qtw.QLabel("Set shortcut for %s" % action))
    kedit = KeySequenceEdit(qtg.QKeySequence())
    form.addRow(kedit)

    buttons = qtw.QDialogButtonBox(
        qtw.QDialogButtonBox.Ok | qtw.QDialogButtonBox.Cancel, qtc.Qt.Horizontal
    )
    form.addRow(buttons)

    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)

    if dialog.exec() == qtw.QDialog.Accepted:
        return kedit.keySequence, True
    else:
        return 0, False


# class for capturing user-defined shortcuts -- from https://stackoverflow.com/a/23919177
class KeySequenceEdit(qtw.QLineEdit):
    def __init__(self, keySequence, *args):
        super(KeySequenceEdit, self).__init__(*args)
        self.keySequence = keySequence
        self.setKeySequence(keySequence)

    def setKeySequence(self, keySequence):
        self.keySequence = keySequence
        self.setText(self.keySequence.toString(qtg.QKeySequence.NativeText))

    def keyPressEvent(self, e):
        if e.type() == qtc.QEvent.KeyPress:
            key = e.key()

            if key == qtc.Qt.Key_unknown:
                warnings.warn("Unknown key from a macro probably")
                return

            if (
                key == qtc.Qt.Key_Control
                or key == qtc.Qt.Key_Shift
                or key == qtc.Qt.Key_Alt
                or key == qtc.Qt.Key_Meta
            ):
                return

            # check for a combination of user clicks
            modifiers = e.modifiers()
            keyText = e.text()
            # if the keyText is empty than it is a special key like F1, F5, ...

            if modifiers & qtc.Qt.ShiftModifier:
                key += qtc.Qt.SHIFT
            if modifiers & qtc.Qt.ControlModifier:
                key += qtc.Qt.CTRL
            if modifiers & qtc.Qt.AltModifier:
                key += qtc.Qt.ALT
            if modifiers & qtc.Qt.MetaModifier:
                key += qtc.Qt.META

            self.setKeySequence(qtg.QKeySequence(key))

    # --------------------------------------------------------
    # Public Methods
    # --------------------------------------------------------
