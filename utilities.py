import os
import sys
import datetime
import time
import inspect
import warnings
import traceback
import hashlib
import zlib
import zipfile
import pickle
import shutil
import re
import logging
import subprocess
from collections import OrderedDict
from collections import namedtuple
from pathlib import PurePath
import pandas as pd
from subprocess import Popen, PIPE

print("python exe: {0}".format(sys.executable))

__version__ = "0.1.1"


# Force logger.warning() to omit the source code line in the message
# formatwarning_orig = warnings.formatwarning
# warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
#    formatwarning_orig(message, category, filename, lineno, line='')

class Utilities(object):
    def __init__(self):
        pass

    colors_txt = OrderedDict()
    colors_txt['black'] = "\033[90m"
    colors_txt['red'] = "\033[91m"
    colors_txt["green"] = "\033[92m"
    colors_txt["yellow"] = "\033[93m"
    colors_txt["blue"] = "\033[94m"
    colors_txt["gray"] = "\033[97m"

    colors_bg = OrderedDict()
    colors_bg['black'] = "\033[100m"
    colors_bg["red"] = "\033[101m"
    colors_bg["green"] = "\033[102m"
    colors_bg["yellow"] = "\033[103m"
    colors_bg["blue"] = "\033[104m"
    colors_bg["gray"] = "\033[107m"
    colors_bg["none"] = "\033[107m"

    txt_effects = OrderedDict()
    txt_effects["end"] = "\033[0m"
    txt_effects["bold"] = "\033[1m"
    txt_effects["underline"] = "\033[4m"
    txt_effects["blackback"] = "\033[7m"

    @staticmethod
    def username():
        return username

    @staticmethod
    def os_whoami():
        proc = subprocess.Popen(['whoami'], stdout=subprocess.PIPE)
        out, errs = proc.communicate()
        return (out)

    @staticmethod
    def now():
        return datetime.datetime.now()

    @staticmethod
    def nowshortstr(fmt="%Y%m%d_%H%M%S"):
        now = datetime.datetime.now()
        res = now.strftime(fmt) + "_" + str(now.microsecond % 10000)
        return res

    @staticmethod
    def nowstr(fmt="%Y-%m-%d__%H_%M_%S"):
        return datetime.datetime.now().strftime(fmt)

    @staticmethod
    def color_str(s, txt_color='black', bg_color=None,
                  bold=False, underline=False,
                  verbosity=0):
        '''
        embedd hex codes for color or effects

        Parameters
        ----------
        s: srting to be enhanced
        txt_color: color for text.  e.g. black, red, green, blue
        bg_color: background color
        bold: boolean
        underline: boolean
        verbosity: level of diagnostics

        Returns
        -------
        string with original and enhancements at the start
        '''
        if verbosity > 0:
            print("{0} <{1}>".format(Utilities.whoami(), Utilities.now()))
        if not isinstance(s, str):
            msg0 = "input s must be string, got {0}".format(type(s))
            msg0 += "trying to convert to string"
            msg = Utilities.color_str(msg0, txt_color="red")
            print(msg)
        try:
            s = str(s)
        except Exception as e:
            msg2 = Utilities.color_str(str(e), txt_color="red", bg_color="red")
            print(msg2)
            raise RuntimeError(msg2)
        result = ''
        if txt_color:
            txt_color = txt_color.lower()
            if txt_color not in Utilities.colors_txt.keys():
                warnings.warn("txt_color '{0}' not a valid color".format(txt_color))
                txt_color = 'black'
        else:
            txt_color = 'black'
        result += Utilities.colors_txt[txt_color]
        if bg_color:
            bg_color = bg_color.lower()
            if bg_color not in Utilities.colors_bg.keys():
                warnings.warn("bg_color '{0}' not a valid color".format(txt_color))
                bg_color = 'none'
        else:
            bg_color = 'none'
        result += Utilities.colors_bg[bg_color]
        if bold:
            result += Utilities.txt_effects['bold']
        if underline:
            result += Utilities.txt_effects['underline']
        result += s + Utilities.txt_effects['end']
        return result

    @staticmethod
    def last_exception_parts():
        (extype, exval, tb) = sys.exc_info()
        return extype, exval, tb

    @staticmethod
    def last_exception_info(verbose=0):
        '''
        returns a string with info about the last exception
        :param verbose:
        :return: string with info about the last exception
        '''
        if verbose > 0:
            print("{0} {1}".format(Utilities.whoami(), Utilities.now()))
        msg = "Exception {0}".format(datetime.datetime.now())
        (extype, exval, tb) = sys.exc_info()
        msg += "\n {0}  type: {1}".format(str(exval), extype)
        tblist = traceback.extract_tb(tb, limit=None)
        lines = traceback.format_list(tblist)
        for i, line in enumerate(lines):
            msg += "\n[{0}] {1}".format(i, line)
        result = Utilities.color_str(msg, txt_color="red")
        return result

    @staticmethod
    def drives(verbosity=0):
        raise RuntimeError("No longer supported")
        fields = ["drive", "dname", "message"]
        DriveTup = namedtuple("DriveTup", fields)
        dlist = []
        drive_strings = None  # win32api.GetLogicalDriveStrings()
        drives = drive_strings.split('\000')[:-1]
        for drive in drives:
            dname = None
            msg = ''
            try:
                dname = None  # win32api.GetVolumeInformation(drive)[0]
            except Exception as e:
                msg = str(e)
            dt = DriveTup(drive, dname, msg)
            dlist.append(dt)
        df = pd.DataFrame(dlist)
        df.columns = fields
        return df

    @staticmethod
    def module_versions(verbosity=0):
        if verbosity > 0:
            print("{0} {1}".format(Utilities.whoami(), Utilities.now()))
        mlist = list(filter(lambda x: inspect.ismodule(x[1]), globals().items()))
        if verbosity > 0:
            print(mlist)
        fields = ["filename", "asname", "ver"]
        ModTup = namedtuple("ModTup", fields)
        tlist = []
        for asname, mod in mlist:
            fname = asname
            ver = None
            if asname.startswith("__"):
                continue
            if hasattr(mod, "__version__"):
                fname = asname
                if hasattr(mod, "__path__"):
                    fname = os.path.split(mod.__path__[0])[1]
                ver = mod.__version__
            mt = ModTup(fname, asname, ver)
            tlist.append(mt)
        df = pd.DataFrame(tlist)
        df.columns = fields
        return df

    @staticmethod
    def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
        return ' %s:%s: %s:%s' % (filename, lineno, category.__name__, message)

    @staticmethod
    def whoami():
        return sys._getframe(1).f_code.co_name

    @staticmethod
    def is_file_binary(filepath,
                       nlines=1000,
                       verbosity=0):
        try:
            with open(filepath, "r") as f:
                for _ in range(nlines):
                    line = f.readline()
                    line.lower()
        except UnicodeDecodeError:
            return True
        return False

    @staticmethod
    def sha_256(fpath,
                fmode='rb',  # default is text mode
                encoding=None,
                size=4096):
        logger = logging.getLogger(__file__)
        m = hashlib.sha256()
        try:
            lastchunk = None
            fsize = os.path.getsize(fpath)
            with open(fpath, mode=fmode, encoding=encoding) as fp:
                try:
                    chunk = None
                    while True:
                        lastchunk = chunk
                        chunk = fp.read(size)
                        if chunk is None or chunk == b'':
                            break
                        m.update(chunk)
                except Exception as ex:
                    errmsg = "fpath: {0}".format(fpath)
                    errmsg += Utilities.last_exception_info()
                    logger.warning(errmsg)
                    (extype, exval, tb) = sys.exc_info()
                    raise extype(exval)
            return m.hexdigest()
        except PermissionError as pe:
            errmsg = "fpath: {0}".format(fpath)
            errmsg += Utilities.last_exception_info()
            logger.warning(errmsg)
            # if tried text, then try binary
            if fmode == 'r':
                return Utilities.sha_256(fpath, fmode='rb', encoding=None)
            else:
                raise PermissionError(pe)
        except TypeError as te:
            errmsg = "fpath: {0}".format(fpath)
            errmsg += Utilities.last_exception_info()
            logger.warning(errmsg)
            if fmode == 'r':
                # try binary
                return Utilities.sha_256(fpath, fmode='rb', encoding=None)
            raise TypeError(te)
        except OSError as oe:
            errmsg = "fpath: {0}".format(fpath)
            errmsg += Utilities.last_exception_info()
            logger.warning(errmsg)
            OSError(oe)
        except Exception as e:
            errmsg = "fpath: {0}".format(fpath)
            errmsg += Utilities.last_exception_info()
            logger.warning(errmsg)
            (extype, exval, tb) = sys.exc_info()
            raise extype(exval)

    @staticmethod
    def handle_exc(e, rethrow=False):
        msg = Utilities.last_exception_info()
        print(msg)
        if rethrow:
            raise RuntimeError(e)

    @staticmethod
    def create_new_zip(infilepath, zipfilepath,
                       compression=zipfile.ZIP_DEFLATED,
                       compresslevel=zlib.Z_DEFAULT_COMPRESSION,
                       verbosity=0):
        if verbosity > 0:
            print("{0} {1}".format(Utilities.whoami(), Utilities.now()))
        if verbosity > 1:
            print("creating zipfile {0} from {1} <{2}>".format(infilepath, zipfilepath,
                                                               datetime.datetime.now()))
        zf = zipfile.ZipFile(zipfilepath, mode='w', compression=compression,
                             compresslevel=compresslevel)
        try:
            if verbosity > 1:
                print("adding {0}".format(infilepath))
            zf.write(infilepath)
        except Exception as e:
            zf.close()
            msg = "infilepath= {0}".format(infilepath)
            msg += Utilities.last_exception_info()
            print(msg)
            raise RuntimeError(msg)
        finally:
            if verbosity > 1:
                print('Done, closing <{0}>'.format(datetime.datetime.now()))
            zf.close()
        return zf

    @staticmethod
    def path2string(fpath, sep="_", verbosity=0):
        if verbosity > 0:
            print("{0} {1}".format(Utilities.whoami(), Utilities.now()))
        pathstring = ""
        pathleft = fpath
        while True:
            pathleft, tail = os.path.split(pathleft)
            if len(tail) == 0:
                break
            pathstring = tail + sep + pathstring
        if verbosity > 0:
            print("pathstring= {0}".format(pathstring))
        return pathstring

    @staticmethod
    def check_outdir(outdir, create=True, verbosity=0):
        if verbosity > 0:
            print("{0} {1}".format(Utilities.whoami(), Utilities.now()))
        if os.path.isdir(outdir):
            return outdir

        warnings.warn("{0} not a dir".format(outdir))
        if not create:
            return None

        if verbosity > 0:
            print("trying to create {0}".format(outdir))
        os.makedirs(outdir)
        if not os.path.isdir(outdir):
            raise RuntimeError("Cannot make dir= '{0}'".format(outdir))
        return outdir

    @staticmethod
    def make_metafilepath(outdir, basename="generic",
                          sep="_", ext="",
                          verbosity=0):
        # Figure out the filename this code should used based on
        # what files already exist.
        if verbosity > 0:
            print("{0} {1}".format(Utilities.whoami(), Utilities.now()))
        while True:
            outfilename = basename + sep + Utilities.nowshortstr() + ext
            if not os.path.exists(outfilename):
                break
        if verbosity > 0:
            print("Creating '{0}'".format(outfilename))

        outfilepath = os.path.join(outdir, outfilename)
        return outfilepath

    @staticmethod
    def make_tempfilepath(folder, base, sep="_", ext="",
                          max_attempts=3,
                          exist_ok=True,
                          verbosity=0):
        if verbosity > 1:
            print("{0} {1}".format(Utilities.whoami(), Utilities.now()))
            print("folder len {0}, folner name: {1}".format(len(folder), folder))
        filepath = None
        if not os.path.isdir(folder):
            if verbosity > 0:
                print("trying to make folder {0}".format(folder))
            try:
                os.makedirs(folder, exist_ok=exist_ok)
            except FileNotFoundError as fe:
                msg = Utilities.last_exception_info()
                warnings.warn(msg)
                raise FileNotFoundError(fe)
            except Exception as e:
                msg = Utilities.last_exception_info()
                warnings.warn(msg)
                raise RuntimeError(e)
        attempt = 0
        while attempt < max_attempts:
            filename = base + sep + Utilities.nowshortstr() + ext
            filepath = os.path.join(folder, filename)
            if len(filepath) > 250:
                logger = logging.getLogger(__file__)
                msg = "filepath len= {0}".len(filepath)
                msg += "\n base= {0}".format(base)
                base = re.sub(" ", "", base)
                msg += "newbase= {0}".format(base)
                logger.warning(msg)
                continue
            if not os.path.exists(filepath):
                break
            attempt += 1
        return filepath

    @staticmethod
    def import_backup_metafile(folder, filename, verbosity=0):
        if verbosity > 0:
            print("{0} {1}".format(Utilities.whoami(), Utilities.now()))
        filepath = os.path.join(folder, filename)
        if not os.path.isfile(filepath):
            raise ValueError("Cannot find file {0} in folder {1}".format(filename, folder))
        data = []
        with open(filepath, "rb") as fp:
            while True:
                try:
                    x = pickle.load(fp)
                    data.append(x)
                except EOFError:
                    # this is expected
                    break
                except Exception as e:
                    Utilities.handle_exc(e)
        return data

    @staticmethod
    def check_folder_filename(folder, filename, verbosity=0):
        if verbosity > 0:
            print("{0} {1}".format(Utilities.whoami(), Utilities.now()))
        filepath = os.path.join(folder, filename)
        if not os.path.isfile(filepath):
            raise ValueError("Cannot find file {0} in folder {1}".format(filename, folder))
        meta = Utilities.import_backup_metafile(folder=folder, filename=filename)
        if len(meta) == 0:
            warnings.warn("Empty metafile {0} in {1}".format(filename, folder))
            return False
        return True

    @staticmethod
    def get_meta(folder, filename, verbosity=0):
        if verbosity > 0:
            print("{0} {1}".format(Utilities.whoami(), Utilities.now()))
        if not Utilities.check_folder_filename(folder, filename):
            return False

        meta = Utilities.import_backup_metafile(folder=folder, filename=filename)
        if len(meta) == 0:
            warnings.warn("Empty metafile {0} in {1}".format(filename, folder))
            return None

        if not meta[0]['rec_type'] == "meta_info":
            msg = "file= {0}, folder= {1}\n first elem is not meta {2}".format(filename, folder, meta[0])
            warnings.warn(msg)
            return None
        return meta

    @staticmethod
    def get_meta_fields(folder, filename):
        if not Utilities.check_folder_filename(folder, filename):
            return False

        meta = Utilities.get_meta(folder, filename)
        if not meta:
            return None

        res = {"meta_info": list(meta[0].keys())}
        if len(meta) > 1:
            res["file_info"] = list(meta[1].keys())
        return res

    @staticmethod
    def get_meta_info(folder, filename, meta_fields=None,
                      file_info_fields=None, verbosity=0):
        if not Utilities.check_folder_filename(folder, filename):
            return False

        meta = Utilities.get_meta(folder, filename)
        if not meta:
            return None
        result = ""
        act_fields = Utilities.get_meta_fields(folder, filename)
        fields = []
        if meta_fields:
            for f in meta_fields:
                if f in act_fields['meta_info']:
                    fields.append(f)
                else:
                    warnings.warn(" requested meta_field {0} not in meta_fields".format(f))
        else:
            fields = act_fields['meta_info']

        msglst = ["{0}: {1}".format(f, meta[0][f]) for f in fields]
        result += ", ".join(msglst)
        result += "\n"

        nfiles = sum([int(e['rec_type'] == 'file_info') for e in meta])
        result += "{0} files".format(nfiles)
        result += "\n"

        fields = []
        if file_info_fields:
            for f in file_info_fields:
                if f in act_fields['file_info']:
                    fields.append(f)
                else:
                    warnings.warn(" requested file_info_field {0} not in file_info_fields".format(f))
        else:
            fields = act_fields['file_info']

        for i, elem in enumerate(meta[1:]):
            msglst = ["[{0}]: {1}: {2}".format(i, f, elem[f]) for f in fields]
            result += ", ".join(msglst)
            result += "\n"
        return result

    @staticmethod
    def check_make_path(thepath, verbosity=0):
        if os.path.isdir(thepath):
            return thepath

        warnings.warn("{0} not a dir".format(thepath))

        if verbosity > 0:
            print("trying to create {0}".format(thepath))

        os.makedirs(thepath)
        if not os.path.isdir(thepath):
            raise RuntimeError("Cannot make dir= '{0}'".format(thepath))

        return thepath

    @staticmethod
    def is_iterable(obj):
        try:
            obj = iter(obj)
            return True
        except:
            return False

    @staticmethod
    def check_folders(folders):
        if isinstance(folders, str):
            folders = [folders]
        elif not Utilities.is_iterable(folders):
            msg = "folders is type {0}, not iterable".format(type(folders))
            raise ValueError(msg)
        errmsg = ''
        for folder in folders:
            if not os.path.isdir(folder):
                errmsg += "'{0}' is not a dir".format(folder)
        if len(errmsg) > 0:
            raise ValueError(errmsg)
        return True

    @staticmethod
    def unzip_to_temp(zipfilepath,
                      tempfolder=None,
                      tempname="temp",
                      verbosity=0):
        if verbosity > 0:
            ldict = locals()
            msg = "{0} <{1}>".format(Utilities.whoami(), Utilities.now())
            for key in ldict.keys():
                print("{0}: {1}".format(key, ldict[key]))

        if tempfolder is None:
            tempfolder = os.path.split(zipfilepath)[0]
        zfile = zipfile.ZipFile(zipfilepath, mode='r')
        zpath = os.path.split(zipfilepath)[0]
        while True:
            tempname = tempname + Utilities.nowshortstr()
            temppath = os.path.join(zpath, tempname)
            if not os.path.isfile(temppath):
                break
            else:
                msg = "Found temp file {0} in {1}\n try another".format(tempname, zpath)
        zinfolist = zfile.infolist()
        if len(zinfolist) != 1:
            zlen = len(zinfolist)
            msg = "file = {0}, zinfolist len= {1}, should be 1".format(zipfilepath, zlen)
            raise ValueError(msg)
        zinfo = zinfolist[0]
        zipname = zinfo.filename
        try:
            if verbosity > 0:
                print("zipname= {0} ".format(zipname))
            zfile.extract(member=zipname,
                          path=temppath, pwd=None)
        except Exception as e:
            Utilities.last_exception_info()
            raise Exception(e)
        finally:
            zfile.close()
        return temppath

