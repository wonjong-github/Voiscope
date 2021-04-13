from gccNMF.realtime.Voiscope import Voiscope
# import os
# import sys
# import kivy
#
# def resourcePath():
#     '''Returns path containing content - either locally or in pyinstaller tmp file'''
#     if hasattr(sys, '_MEIPASS'):
#         return os.path.join(sys._MEIPASS)
#
#     return os.path.join(os.path.abspath("."))


if __name__ =='__main__':
    # kivy.resources.resource_add_path(resourcePath())  # add this line
    Voiscope().run()
