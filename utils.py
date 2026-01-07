from os import path, makedirs

plotly_show_config = {
    'toImageButtonOptions': {
        'format': 'svg',
        'filename': 'unset',
        'width': 800, 'height': 450
    },
    "editable": True,
}

def plotly_export(fig, filename, html=False, show=False, **kwargs):
    dir = path.dirname(f'./plots/{filename}')
    if not path.exists(dir):
        makedirs(dir)
    fig.update_layout(margin=dict(t=50, b=0, l=0, r=0), width=800, height=450)
    fig.write_image(f"./plots/{filename}.eps", width=500, height=300,format='eps', engine='kaleido')
    plotly_show_config['toImageButtonOptions']['filename'] = filename
    if html:
        fig.write_html(f"./plots/{filename}.html", config=plotly_show_config)
    if show:
        fig.show(config=plotly_show_config)
    plotly_show_config['toImageButtonOptions']['filename'] = 'unset'
