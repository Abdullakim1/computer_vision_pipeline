"""CineForge generative video synthesis studio."""

from dotenv import load_dotenv
load_dotenv()

from .types import GeneratedClip, GenerationRequest
from .studio import CineForgeStudio

__version__ = "1.0.0"
__author__ = "Abdullakim"