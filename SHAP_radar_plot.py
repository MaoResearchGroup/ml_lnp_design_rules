import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo


def plot_Radar(cell_type, features, values, save_path):

    features = [*features, features[0]]

    values = [*values, values[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(r=values, theta=features, fill='toself', name= cell_type),

        ],
        layout=go.Layout(
            title=go.layout.Title(text=cell_type),
            polar={'radialaxis': {'visible': True}},
            showlegend=True,
                font=dict(size=24)
        )
    )
    fig.show()
    #pyo.plot(fig)

    #fig.to_image(format="png", engine="kaleido")
    # fig.write_image(save_path + f"{cell_type}_radar.png")
    # print(save_path + f"{cell_type}_radar.png")
def main():
    ################ Retreive/Store Data ##############################################
    save_path = "Figure/Radar/"
    features = ['Dlin-MC3:Helper Lipid Ratio', 
                'Dlin-MC3 + Helper Lipid Percentage', 
                'NP Ratio', 
                'Cholesterol:DMG-PEG Ratio', 
                'Helper Lipid cLogP']

    ################ INPUT PARAMETERS ############################################
    #cell_type_names = ['HEK293','HepG2', 'N2a', 'ARPE19', 'B16', 'PC3']
    cell_type_names = ['HEK293','HepG2', 'N2a', 'ARPE19', 'B16', 'PC3', 'overall']
    for cell in cell_type_names:
        if cell == "B16":
            values = [1, 6, 1, 10, 7]
        elif cell =="HEK293":
            values = [1, 7, 10, 3, 10]
        elif cell == "N2a":
            values = [1, 1, 3, 3, 7]
        elif cell == "HepG2":
            values = [1, 10, 10, 3, 10]
        elif cell == "ARPE19":
            values = [1, 7, 10, 3, 10]
        elif cell == "PC3":
            values = [1, 1, 10, 3, 10]
        elif cell =="overall":
            values = [1, 7, 10, 3, 7]
        else:
            values = [0, 0, 0, 0, 0]
        plot_Radar(cell, features, values, save_path)


if __name__ == "__main__":
    main()