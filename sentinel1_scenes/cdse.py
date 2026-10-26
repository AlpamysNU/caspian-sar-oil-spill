"""CDSE access: Copernicus Data Space S3 / OData

    python -m sentinel1_scenes.cdse probe <scene .SAFE path under eodata/> <out_dir>       (S3 keys from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, never stored)
    python -m sentinel1_scenes.cdse annotation <eodata/.../SCENE.SAFE> <out_dir>             (VV annotation, calibration and noise XML only)
    python -m sentinel1_scenes.cdse download <product_id> <out_dir>                           (OData; CDSE e-mail and password are asked, never stored)
"""


import getpass
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

from segmentation import paths



# ----------------------------------------------------------------------------- probe


DOC_probe = """Probe one Sentinel-1 COG scene on Copernicus Data Space S3 and copy its VV layer locally.

Keys are read from the environment of this single run and never written anywhere:
  AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... python3 probe_s3.py <scene .SAFE path under eodata/> <out_dir>
Uses the system GDAL (/vsis3/), so run with the system python3, not the venv."""


WEB = paths.PROJECT_ROOT / 'sentinel1'


def main_probe():
    from osgeo import gdal
    if not (os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY')):
        sys.exit('set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in the environment of this run')
    scene, out_dir = sys.argv[1].strip('/'), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, value in {'AWS_S3_ENDPOINT': 'eodata.dataspace.copernicus.eu', 'AWS_VIRTUAL_HOSTING': 'FALSE',
                       'AWS_HTTPS': 'YES', 'GDAL_HTTP_MAX_RETRY': '3'}.items():
        gdal.SetConfigOption(key, value)
    gdal.UseExceptions()

    base = '/vsis3/' + scene
    names = gdal.ReadDirRecursive(base + '/')
    if not names:
        sys.exit('listing failed, see messages above')
    print('--- files in scene')
    for n in names:
        print(' ', n, gdal.VSIStatL(f'{base}/{n}').size if not n.endswith('/') else '')
    vv = [n for n in names if 'measurement' in n and '-vv-' in n and n.endswith('.tiff')][0]
    src = f'{base}/{vv}'

    ds = gdal.Open(src)
    band = ds.GetRasterBand(1)
    print('--- VV', ds.RasterXSize, 'x', ds.RasterYSize, band.DataType and gdal.GetDataTypeName(band.DataType),
          'overviews', band.GetOverviewCount(), 'block', band.GetBlockSize())
    print('geotransform', ds.GetGeoTransform(), 'gcps', ds.GetGCPCount())

    t = time.time()
    window = band.ReadAsArray(ds.RasterXSize // 2, ds.RasterYSize // 2, 1250, 650)
    print(f'--- range read 1250x650 window: {time.time() - t:.1f} s, mean {window.mean():.1f}, max {window.max()}')

    target = out_dir / 'vv_full.tif'
    t = time.time()
    def progress(fraction, message, data):
        print(f'\r  copying VV: {fraction * 100:5.1f} %  ({time.time() - t:.0f} s)', end='', flush=True)
        return 1


    gdal.Translate(str(target), src, creationOptions=['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=IF_SAFER'],
                   callback=progress)
    print()
    print(f'--- full VV copied to {target} in {time.time() - t:.0f} s, {target.stat().st_size / 1e6:.0f} MB')



# ----------------------------------------------------------------------------- annotation


DOC_annotation = """Download the small VV annotation, calibration and noise XML files (and the manifest) of one scene from CDSE S3.

Keys come from the environment of this single run and are not stored:
  AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... python3 fetch_annotation.py <eodata/.../SCENE.SAFE> <out_dir>
Files are read with VSIFReadL (no numpy), because ReadAsArray of the system GDAL binding crashes here."""


def main_annotation():
    from osgeo import gdal
    if not (os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY')):
        sys.exit('set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in the environment of this run')
    scene, out = sys.argv[1].strip('/'), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for key, value in {'AWS_S3_ENDPOINT': 'eodata.dataspace.copernicus.eu', 'AWS_VIRTUAL_HOSTING': 'FALSE', 'AWS_HTTPS': 'YES'}.items():
        gdal.SetConfigOption(key, value)
    base = '/vsis3/' + scene
    names = gdal.ReadDirRecursive(base + '/') or sys.exit('listing failed (check keys / path)')
    wanted = [n for n in names if n.endswith('.xml') and ('-vv-' in n or n == 'manifest.safe') or n == 'manifest.safe']
    for n in sorted(set(wanted)):
        f = gdal.VSIFOpenL(f'{base}/{n}', 'rb')
        size = gdal.VSIStatL(f'{base}/{n}').size
        data = gdal.VSIFReadL(1, size, f)
        gdal.VSIFCloseL(f)
        target = out / n
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f'{size:>10} B  {n}')



# ----------------------------------------------------------------------------- download


DOC_download = """Download one Sentinel-1 product from Copernicus Data Space (OData) and unzip it.
Credentials are asked interactively (hidden input) and never stored.
Usage: python download_scene.py <product_id> <out_dir>"""


TOKEN_URL = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'


def main_download():
    product_id, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    user = input('CDSE e-mail: ').strip()
    password = getpass.getpass('CDSE password (hidden): ')
    r = requests.post(TOKEN_URL, data={'client_id': 'cdse-public', 'grant_type': 'password',
                                       'username': user, 'password': password}, timeout=60)
    r.raise_for_status()
    token = r.json()['access_token']
    url = f'https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value'
    zip_path = out_dir / f'{product_id}.zip'
    with requests.get(url, headers={'Authorization': f'Bearer {token}'}, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0)); done = 0
        with open(zip_path, 'wb') as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk); done += len(chunk)
                print(f'\r{done/1e6:8.0f} / {total/1e6:.0f} MB', end='', flush=True)
    print('\nunzipping')
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    zip_path.unlink()
    print('done:', *[p.name for p in out_dir.iterdir()])

COMMANDS = {
    'probe': main_probe,
    'annotation': main_annotation,
    'download': main_download,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print('usage: python -m sentinel1_scenes.cdse <command> [options]\n\ncommands:\n  ' + '\n  '.join(COMMANDS))
        raise SystemExit(0 if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help') else 2)
    command = sys.argv.pop(1)
    sys.argv[0] = f'sentinel1_scenes.cdse {command}'
    COMMANDS[command]()


if __name__ == '__main__':
    main()
