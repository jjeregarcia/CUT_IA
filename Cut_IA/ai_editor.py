"""
ai_editor.py
Ejecuta las acciones de edición sobre el video usando MoviePy y pydub.

Acciones soportadas en el diccionario:
    remove_silence : bool              → elimina segmentos silenciosos
    duration       : int               → recorta el video a N segundos
    speed          : float             → cambia la velocidad (ej: 1.5, 0.75)
    subtitles      : bool              → placeholder faster-whisper (semana 3)
    blackwhite     : (float, float)    → aplica B&N entre segundo X e Y
    zoom           : bool | float      → zoom progresivo (True = 1.3x al final)
    volume         : float             → multiplica el volumen (ej: 1.5 = +50%)
    meme           : dict              → inserta clip en un momento dado
                       "path": str     → ruta al archivo del meme
                       "time": float   → segundo en que se inserta

Ejemplo completo:
    acciones = {
        "remove_silence": True,
        "duration": 30,
        "speed": 1.5,
        "blackwhite": (5, 10),
        "zoom": True,
        "volume": 1.5,
        "meme": {"path": "memes/meme1.mp4", "time": 8}
    }
"""
from faster_whisper import WhisperModel
import os
import tempfile

import numpy as np
from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
)
from moviepy.video.fx.all import blackwhite, speedx
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"})

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def procesar_video(ruta: str, acciones: dict, ruta_salida: str = "uploads/video_editado.mp4") -> str:
    """
    Aplica las acciones del diccionario al video y guarda el resultado.
    Devuelve la ruta del archivo de salida.

    Orden de aplicación (importa para no romper sincronización):
        1. Recortar duración
        2. Cambiar velocidad
        3. Eliminar silencios
        4. Insertar meme
        5. Blanco y negro en segmento
        6. Zoom progresivo
        7. Subir/bajar volumen
        8. Subtítulos (placeholder)
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo de video: {ruta}")
    # Limpiar valores inválidos del diccionario de acciones
    if acciones.get("fade_in") in (None,): acciones["fade_in"] = 1.0
    if acciones.get("fade_out") in (None,): acciones["fade_out"] = 1.0
    if acciones.get("speed") in (None, 1.0):        acciones["speed"] = None
    if acciones.get("volume") in (None, 1.0):       acciones["volume"] = None
    if acciones.get("zoom") in (None, False, 1.0):  acciones["zoom"] = None
    if acciones.get("blackwhite") and (
        not isinstance(acciones["blackwhite"], (list, tuple)) or
        None in acciones["blackwhite"] or
        len(acciones["blackwhite"]) != 2
    ):
        acciones["blackwhite"] = None
    video = VideoFileClip(ruta)
    duracion_real = video.duration
    print(f"📹 Video cargado: {duracion_real:.1f}s — {video.w}x{video.h}px")

    # ── 1. Recortar duración ─────────────────────────────────────────────────
    if acciones.get("duration"):
        limite = min(float(acciones["duration"]), duracion_real)
        video = video.subclip(0, limite)
        print(f"✂️  Recortado a {limite}s")

    # ── 2. Cambiar velocidad ─────────────────────────────────────────────────
    if acciones.get("speed") and acciones["speed"] is not None and acciones["speed"] != 1.0:
        video = cambiar_velocidad(video, acciones["speed"])

    # ── 3. Eliminar silencios ────────────────────────────────────────────────
    if acciones.get("remove_silence"):
        video = eliminar_silencios(video)

    # ── 4. Insertar meme ─────────────────────────────────────────────────────
    if acciones.get("meme"):
        cfg = acciones["meme"]
        video = insertar_meme(video, path=cfg.get("path"), tiempo=cfg.get("time", 0))

    # ── 5. Blanco y negro en segmento ───────────────────────────────────────
    if acciones.get("blackwhite") and None not in acciones["blackwhite"]:
        inicio, fin = acciones["blackwhite"]
        video = aplicar_blanco_y_negro(video, inicio, fin)

    # ── 6. Zoom progresivo ───────────────────────────────────────────────────
    if acciones.get("zoom"):
        # Si zoom=True usa escala máxima 1.3. Si zoom=float, usa ese valor.
        escala_max = acciones["zoom"] if isinstance(acciones["zoom"], float) else 1.3
        video = aplicar_zoom_progresivo(video, escala_max)

    # ── 7. Volumen ───────────────────────────────────────────────────────────
    if acciones.get("volume") and acciones["volume"] != 1.0:
        video = ajustar_volumen(video, acciones["volume"])

    # ── 8. Subtítulos (placeholder) ──────────────────────────────────────────
    if acciones.get("subtitles"):
        video = agregar_subtitulos(video)
    
    # ── 9. Fade in / fade out ─────────────────────────────────────────────
    video = aplicar_fade(video, 
        fade_in=acciones.get("fade_in", 1.0),
        fade_out=acciones.get("fade_out", 1.0)
    )
    # ── 10. Música de fondo ───────────────────────────────────────────────
    if acciones.get("music_path"):
        video = agregar_musica(video, acciones["music_path"], volumen=acciones.get("music_volume", 0.3))

    # ── 11. Highlights ────────────────────────────────────────────────────
    if acciones.get("highlights"):
        duracion_hl = acciones.get("highlights_duration") or 30.0
        video = detectar_highlights(video, duracion_total=duracion_hl)
    # ── Guardar resultado ────────────────────────────────────────────────────
    os.makedirs("uploads", exist_ok=True)
    salida = ruta_salida

    video.write_videofile(
        salida,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="uploads/temp_audio_out.m4a",
        remove_temp=True,
        logger=None,
    )

    video.close()
    print(f"✅ Video guardado en: {salida}")
    return salida


# ============================================================================
# FUNCIONES DE EDICIÓN
# ============================================================================

def cambiar_velocidad(video: VideoFileClip, factor: float) -> VideoFileClip:
    """
    Acelera o desacelera el video.
    factor > 1 → más rápido | factor < 1 → más lento
    El audio también se ajusta (tono puede variar levemente en MoviePy 1.0.3).
    """
    if not (0.1 <= factor <= 10):
        raise ValueError(f"Factor de velocidad inválido: {factor}. Debe estar entre 0.1 y 10.")
    resultado = speedx(video, factor)
    print(f"⚡ Velocidad ajustada a {factor}x")
    return resultado


def eliminar_silencios(
    video: VideoFileClip,
    min_silence_len: int = 700,
    silence_thresh: int = -38,
    padding_ms: int = 150,
) -> VideoFileClip:
    """
    Detecta partes con audio y descarta los silencios.
    padding_ms: margen extra antes/después de cada segmento para evitar cortes bruscos.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        ruta_wav = tmp.name

    try:
        video.audio.write_audiofile(ruta_wav, logger=None)
        audio = AudioSegment.from_wav(ruta_wav)

        segmentos = detect_nonsilent(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
        )

        if not segmentos:
            print("⚠️  No se detectaron segmentos con audio. Video sin cambios.")
            return video

        duracion_ms = len(audio)
        clips = []
        for start_ms, end_ms in segmentos:
            start_ms = max(0, start_ms - padding_ms)
            end_ms   = min(duracion_ms, end_ms + padding_ms)
            clips.append(video.subclip(start_ms / 1000, end_ms / 1000))

        resultado = concatenate_videoclips(clips)
        print(f"🔇 Silencios eliminados: {len(clips)} segmento(s) conservados.")
        return resultado

    finally:
        if os.path.exists(ruta_wav):
            os.remove(ruta_wav)


def aplicar_blanco_y_negro(
    video: VideoFileClip,
    inicio: float,
    fin: float,
) -> VideoFileClip:
    """
    Convierte a blanco y negro solo el segmento entre `inicio` y `fin` (segundos).
    El resto del video mantiene el color original.

    Cómo funciona:
        Se corta el video en 3 partes: antes | segmento B&N | después
        Se concatenan las 3 partes al final.
    """
    duracion = video.duration

    # Validaciones
    if inicio < 0 or fin <= inicio or fin > duracion:
        print(f"⚠️  Rango B&N inválido ({inicio}s-{fin}s) para video de {duracion:.1f}s. Se omite.")
        return video

    partes = []

    # Parte antes del efecto (puede no existir si inicio=0)
    if inicio > 0:
        partes.append(video.subclip(0, inicio))

    # Segmento en blanco y negro
    segmento_bw = blackwhite(video.subclip(inicio, fin))
    partes.append(segmento_bw)

    # Parte después del efecto (puede no existir si fin=duracion)
    if fin < duracion:
        partes.append(video.subclip(fin, duracion))

    resultado = concatenate_videoclips(partes)
    print(f"🎞️  Blanco y negro aplicado de {inicio}s a {fin}s")
    return resultado


def aplicar_zoom_progresivo(
    video: VideoFileClip,
    escala_max: float = 1.3,
) -> VideoFileClip:
    """
    Aplica un zoom suave que va de 1.0x al inicio hasta `escala_max` al final.
    El video se recorta al centro para mantener las dimensiones originales.

    escala_max = 1.3 → zoom del 30% al final del clip.
    """
    if escala_max <= 1.0:
        print("⚠️  escala_max debe ser mayor a 1.0 para que el zoom sea visible.")
        return video

    w_orig, h_orig = video.w, video.h
    duracion = video.duration

    def zoom_frame(get_frame, t):
        """
        Función que se aplica cuadro a cuadro.
        t → tiempo actual en segundos.
        """
        frame = get_frame(t)

        # Escala lineal: de 1.0 al inicio hasta escala_max al final
        progreso = t / duracion                            # 0.0 → 1.0
        escala   = 1.0 + (escala_max - 1.0) * progreso   # ej: 1.0 → 1.3

        # Nuevo tamaño del cuadro escalado
        nuevo_w = int(w_orig * escala)
        nuevo_h = int(h_orig * escala)

        # Redimensionar usando interpolación simple con numpy
        # (MoviePy 1.0.3 usa PIL internamente en resize)
        from PIL import Image
        img = Image.fromarray(frame)
        img = img.resize((nuevo_w, nuevo_h), Image.LANCZOS)
        frame_grande = np.array(img)

        # Recortar al centro para mantener w_orig x h_orig
        x0 = (nuevo_w - w_orig) // 2
        y0 = (nuevo_h - h_orig) // 2
        frame_recortado = frame_grande[y0:y0 + h_orig, x0:x0 + w_orig]

        return frame_recortado

    resultado = video.fl(zoom_frame, apply_to=["mask"])
    print(f"🔍 Zoom progresivo aplicado: 1.0x → {escala_max}x")
    return resultado


def ajustar_volumen(video: VideoFileClip, factor: float) -> VideoFileClip:
    """
    Multiplica el volumen del audio por `factor`.
    factor > 1 → más fuerte | factor < 1 → más suave | factor = 0 → silencio
    """
    if factor < 0:
        raise ValueError("El factor de volumen no puede ser negativo.")
    if video.audio is None:
        print("⚠️  El video no tiene audio. Se omite ajuste de volumen.")
        return video

    resultado = video.volumex(factor)
    resultado.fps = video.fps
    print(f"🔊 Volumen ajustado a {factor}x")
    return resultado


def insertar_meme(
    video: VideoFileClip,
    path: str,
    tiempo: float,
) -> VideoFileClip:
    """
    Inserta un clip corto (meme) en el momento `tiempo` del video principal.

    Cómo funciona:
        video principal = [parte_A] + [meme] + [parte_B]
        El meme se redimensiona al tamaño del video principal automáticamente.

    path   → ruta al archivo del meme (mp4, gif, etc.)
    tiempo → segundo del video principal donde se inserta
    """
    if not path or not os.path.exists(path):
        print(f"⚠️  Archivo de meme no encontrado: '{path}'. Se omite.")
        return video

    duracion = video.duration

    if tiempo < 0 or tiempo >= duracion:
        print(f"⚠️  Tiempo de inserción {tiempo}s fuera del rango del video ({duracion:.1f}s). Se omite.")
        return video

    try:
        meme = VideoFileClip(path)
    except Exception as e:
        print(f"⚠️  No se pudo cargar el meme: {e}. Se omite.")
        return video

    # Redimensionar meme al tamaño del video principal
    if meme.size != video.size:
        meme = meme.resize(video.size)

    # Parte antes del meme
    parte_antes = video.subclip(0, tiempo)

    # Parte después del meme
    parte_despues = video.subclip(tiempo, duracion)

    resultado = concatenate_videoclips([parte_antes, meme, parte_despues])
    print(f"😂 Meme insertado en el segundo {tiempo}s (duración del meme: {meme.duration:.1f}s)")
    return resultado


def agregar_subtitulos(video: VideoFileClip) -> VideoFileClip:
    import tempfile
    from moviepy.editor import TextClip, CompositeVideoClip

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        ruta_wav = tmp.name

    try:
        video.audio.write_audiofile(ruta_wav, logger=None)

        print("🎙️  Transcribiendo audio con Whisper...")
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(ruta_wav, language="es")

        subs = []
        for seg in segments:
            subs.append(((seg.start, seg.end), seg.text.strip()))

        if not subs:
            print("⚠️  No se detectó texto para subtitular.")
            return video

        def make_text_clip(txt):
            return TextClip(
                txt,
                fontsize=38,
                color="white",
                stroke_color="black",
                stroke_width=1.5,
                method="caption",
                size=(int(video.w * 0.85), None),
                font="Arial",
            ).set_position(("center", 0.82), relative=True)

        sub_clips = []
        for (start, end), txt in subs:
            clip = make_text_clip(txt).set_start(start).set_end(end)
            sub_clips.append(clip)
        print(f"DEBUG fps: {video.fps}, audio: {video.audio}")
        resultado = CompositeVideoClip([video] + sub_clips)
        resultado = resultado.set_audio(video.audio)
        resultado.fps = video.fps
        print(f"💬 Subtítulos agregados: {len(subs)} segmento(s).")
        return resultado

    finally:
        if os.path.exists(ruta_wav):
            os.remove(ruta_wav)

def aplicar_fade(video: VideoFileClip, fade_in: float = 1.0, fade_out: float = 1.0) -> VideoFileClip:
    """
    Aplica fade in al inicio y fade out al final del video.
    fade_in / fade_out → duración en segundos de cada efecto.
    """
    from moviepy.video.fx.all import fadein, fadeout
    from moviepy.audio.fx.all import audio_fadein, audio_fadeout

    if fade_in > 0:
        video = fadein(video, fade_in)
        if video.audio:
            video = video.set_audio(audio_fadein(video.audio, fade_in))

    if fade_out > 0:
        video = fadeout(video, fade_out)
        if video.audio:
            video = video.set_audio(audio_fadeout(video.audio, fade_out))

    print(f"🎬 Fade aplicado: in={fade_in}s, out={fade_out}s")
    return video

def agregar_musica(video: VideoFileClip, ruta_musica: str, volumen: float = 0.3) -> VideoFileClip:
    """
    Mezcla una pista de música de fondo con el audio original del video.
    volumen → nivel de la música (0.3 = 30% para no tapar el audio original)
    """
    from moviepy.editor import AudioFileClip
    from moviepy.audio.fx.all import audio_loop

    if not os.path.exists(ruta_musica):
        print(f"⚠️  Archivo de música no encontrado: {ruta_musica}. Se omite.")
        return video

    musica = AudioFileClip(ruta_musica).volumex(volumen)

    # Si la música es más corta que el video, la repite en loop
    if musica.duration < video.duration:
        musica = audio_loop(musica, duration=video.duration)
    else:
        musica = musica.subclip(0, video.duration)

    # Mezclar audio original con música
    if video.audio:
        from moviepy.audio.AudioClip import CompositeAudioClip
        audio_final = CompositeAudioClip([video.audio, musica])
    else:
        audio_final = musica

    resultado = video.set_audio(audio_final)
    resultado.fps = video.fps
    print(f"🎵 Música de fondo agregada: {ruta_musica} al {int(volumen*100)}% de volumen")
    return resultado

def detectar_highlights(video: VideoFileClip, duracion_total: float = 30.0, ventana: float = 3.0) -> VideoFileClip:
    """
    Detecta los momentos más energéticos del video y arma un clip resumen.
    duracion_total → duración aproximada del highlight en segundos.
    ventana        → duración de cada segmento recortado en segundos.
    """
    import tempfile
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        ruta_wav = tmp.name

    try:
        video.audio.write_audiofile(ruta_wav, logger=None)
        audio = AudioSegment.from_wav(ruta_wav)

        # Dividir en ventanas y calcular energía de cada una
        ventana_ms = int(ventana * 1000)
        energias = []
        for i in range(0, len(audio) - ventana_ms, ventana_ms // 2):
            segmento = audio[i:i + ventana_ms]
            energia  = segmento.rms
            tiempo   = i / 1000
            energias.append((energia, tiempo))

        if not energias:
            print("⚠️  No se pudo analizar el audio para highlights.")
            return video

        # Ordenar por energía descendente y tomar los mejores
        energias.sort(reverse=True)
        n_segmentos = max(1, int(duracion_total / ventana))
        mejores = sorted(energias[:n_segmentos], key=lambda x: x[1])  # reordenar por tiempo

        # Evitar solapamientos
        clips = []
        ultimo_fin = -ventana
        for energia, inicio in mejores:
            if inicio >= ultimo_fin:
                fin = min(inicio + ventana, video.duration)
                clips.append(video.subclip(inicio, fin))
                ultimo_fin = fin

        if not clips:
            print("⚠️  No se detectaron highlights.")
            return video

        resultado = concatenate_videoclips(clips)
        print(f"⭐ Highlights detectados: {len(clips)} segmento(s) — {resultado.duration:.1f}s total")
        return resultado

    finally:
        if os.path.exists(ruta_wav):
            os.remove(ruta_wav)