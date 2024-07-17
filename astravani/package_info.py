MAJOR = 0
MINOR = 0
PATCH = 0
PRE_RELEASE = "rc0"

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = ".".join(map(str, VERSION[:3]))
__version__ = ".".join(map(str, VERSION[:3])) + "".join(VERSION[3:])

__package_name__ = "astravani"
__contact_names__ = "Ashwin Sankar"
__contact_emails__ = "ashwins1211@gmail.com"
__repository_url__ = "https://github.com/iamunr4v31/astravani"
__description__ = "Weapons to wield audio and speech"
__license__ = "Apache2"
__keywords__ = "astravani, pytorch, torch, tts, speech, language, speech synthesis"