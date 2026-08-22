"""CineForge - Generative Video Synthesis Studio.

Transforms your CV pipeline into a production-graded video generation
platform. Vision stack provides the conditioning layer; the studio offers
multi-backend synthesis (cloud + local + procedural cinematic).

>>> from src.forge import CineForgeStudio
>>> studio = CineForgeStudio(seed=42)
>>> clip = studio.text_to_video("aurora over a mountain valley")
>>> clip.write_video("demo.mp4")
"""

__version__ = "1.0.0"
__author__ = "Abdullakim"
