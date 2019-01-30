import click
import json
import requests
import gettext
from ltldoorstep import printer
from ltldoorstep.config import load_config
from ltldoorstep.engines import engines
from ltldoorstep.file import make_file_manager
import asyncio
import logging

# TODO: possibly replace with e.g. dynaconf as needs evolve
def get_engine(engine, config):
    if ':' in engine:
        engine, engine_options = engine.split(':')
        sp = lambda x: (x.split('=') if '=' in x else (x, True))
        engine_options = {k: v for k, v in map(sp, engine_options.split(','))}
    else:
        engine_options = {}

    if 'engine' not in config:
        config['engine'] = {}

    for option, value in engine_options.items():
        config['engine'][option] = value

    return engine, config

@click.group()
@click.option('--debug/--no-debug', default=False)
@click.option('-b', '--bucket', default=None)
@click.option('-o', '--output', type=click.Choice(printer.get_printer_types()), default='ansi')
@click.option('--output-file', default=None)
@click.pass_context
def cli(ctx, debug, bucket, output, output_file):
    gettext.install('ltldoorstep')

    prnt = printer.get_printer(output, debug, target=output_file)

    config = load_config()

    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
    logger = logging.getLogger(__name__)

    ctx.obj = {
        'DEBUG': debug,
        'printer': prnt,
        'config': config,
        'bucket': bucket,
        'logger': logger
    }

@cli.command(name='engine-info')
@click.argument('engine', 'engine to get information about', required=False)
@click.pass_context
def engine_info(ctx, engine=None):
    if engine:
        if engine in engines:
            click.echo(_('Engine details: %s') % engine)
            click.echo(_('    %s') % engines[engine].description())
            click.echo()
            config_help = engines[engine].config_help()
            if config_help:
                for setting, description in config_help.items():
                    click.echo("%s:\n\t%s" % (setting, description.replace('\n', '\n\t')))
            else:
                click.echo("No configuration settings for this engine")
        else:
            click.echo(_('Engine not known'))
    else:
        click.echo(_('Engines available:') + ' ' + ', '.join(engines))

@cli.command()
@click.pass_context
def status(ctx):
    debug = ctx.obj['DEBUG']
    click.echo(_('STATUS'))

    if debug:
        click.echo(_('Debug is on'))
    else:
        click.echo(_('Debug is off'))

@cli.command()
@click.argument('filename', 'data file to process')
@click.argument('workflow', 'Python workflow module')
@click.option('-e', '--engine', required=True)
@click.option('-m', '--metadata', default=None)
@click.pass_context
def process(ctx, filename, workflow, engine, metadata):
    printer = ctx.obj['printer']
    config = ctx.obj['config']
    bucket = ctx.obj['bucket']

    engine, config = get_engine(engine, config)
    click.echo(_("Engine: %s" % engine))
    engine = engines[engine](config=config)

    if metadata is None:
        metadata = {}
    else:
        with open(metadata, 'r') as metadata_file:
            metadata = json.load(metadata_file)

    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(engine.run(filename, workflow, metadata, bucket=bucket))
    printer.build_report(result)

    printer.print_output()

@cli.command()
@click.option('--engine', required=True)
@click.option('--protocol', type=click.Choice(['http', 'wamp']), required=True)
@click.option('--router', default='localhost:8080')
@click.pass_context
def serve(ctx, engine, protocol, router):
    printer = ctx.obj['printer']
    config = ctx.obj['config']

    engine, config = get_engine(engine, config)
    click.echo(_("Engine: %s" % engine))
    engine = engines[engine](config=config)

    if protocol == 'http':
        from ltldoorstep.flask_server import launch_flask
        launch_flask(engine)
    elif protocol == 'wamp':
        from ltldoorstep.wamp_server import launch_wamp
        launch_wamp(engine, router, config)
    else:
        raise RuntimeError(_("Unknown protocol"))

@cli.command()
@click.argument('workflow', 'Python workflow module')
@click.option('--url', required=True)
@click.option('--engine', default='dask.threaded')
@click.pass_context
def crawl(ctx, workflow, url, engine):
    printer = ctx.obj['printer']
    config = ctx.obj['config']

    engine, config = get_engine(engine, config)

    click.echo(_("Engine: %s" % engine))
    engine = engines[engine](config=config)

    metadata = {}

    from ckanapi import RemoteCKAN
    client = RemoteCKAN(url, user_agent='lintol-doorstep-crawl/1.0 (+http://lintol.io)')
    resources = client.action.resource_search(query='format:csv')
    if 'results' in resources:
        for resource in resources['results']:
            r = requests.get(resource['url'])
            with make_file_manager(content={'data.csv': r.text}) as file_manager:
                filename = file_manager.get('data.csv')
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(engine.run(filename, workflow, metadata))
                printer.build_report(result)
    printer.print_output()
