DOCK_TITLE_STYLESHEET = '''
    * {
        padding: 0;
        margin: 0;
        border: 0;
        background: #232426;
        color: white;
    }
    QPushButton {
        padding: 4px;
        margin: 0 1px;
    }
    QCheckBox {
        padding: 2px 4px;
        margin: 0 1px;
    }
    QLabel {
        padding: 3px;
        font-size: 12px;
    }
    QPushButton:hover, QCheckBox:hover {
        background: #323438;
    }
    QPushButton:pressed {
        background: #53575e;
    }
    QPushButton:checked {
        background: #6c717a;
    }
'''

DOCK_LABEL_STYLESHEET = """
    DockLabel {
        background-color : #232426;
        color : white;
        border-top-right-radius: 0px;
        border-top-left-radius: 0px;
        border-bottom-right-radius: 0px;
        border-bottom-left-radius: 0px;
        border-width: 0px;
        padding-left: 3px;
        padding-right: 3px;
        font-size: 12px;
        qproperty-alignment: 'AlignVCenter | AlignLeft';
        qproperty-wordWrap: true;
    }

    QPushButton {
        padding: 4px;
        margin: 0 1px;
    }

    QCheckBox {
        padding: 2px 4px;
        margin: 0 1px;
    }

    QLabel {
        padding: 3px;
        font-size: 12px;
    }

    QPushButton:hover, QCheckBox:hover {
        background: #323438;
    }

    QPushButton:pressed {
        background: #53575e;
    }

    QPushButton:checked {
        background: #6c717a;
    }
"""

DOCK_LABEL_DIM_STYLESHEET = """
    DockLabel {
        background-color : grey;
        color : white;
        border-top-right-radius: 0px;
        border-top-left-radius: 0px;
        border-bottom-right-radius: 0px;
        border-bottom-left-radius: 0px;
        border-width: 0px;
        padding-left: 3px;
        padding-right: 3px;
        font-size: 12px;
        qproperty-alignment: 'AlignVCenter | AlignLeft';
        qproperty-wordWrap: true;

    }

    QPushButton {
        padding: 4px;
        margin: 0 1px;
    }

    QCheckBox {
        padding: 2px 4px;
        margin: 0 1px;
    }

    QLabel {
        padding: 3px;
        font-size: 12px;
    }

    QPushButton:hover, QCheckBox:hover {
        background: #323438;
    }

    QPushButton:pressed {
        background: #53575e;
    }

    QPushButton:checked {
        background: #6c717a;
    }
"""

DOCK_STATUS_STYLESHEET = '''
    * {
        padding: 0;
        margin: 0;
        border: 0;
        background: black;
        color: white;
    }

    QLabel {
        padding: 3px;
    }
'''