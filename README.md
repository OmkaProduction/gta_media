# GTA VI clips

Automatyczne cięcie klipów 9:16 ze streamów GTA VI dla kampanii clippingowych (Content Rewards).

## Jak to działa

Co 3 godziny GitHub Actions (`.github/workflows/clips.yml`):

1. Sprawdza nowe, zakończone VOD-y streamerów z `config/streamers.yaml` (Twitch, Kick, YouTube).
2. Pobiera samo audio i szuka najgłośniejszych reakcji (śmiech, krzyk) względem tła streamu.
3. Robi transkrypcję tych momentów (Whisper).
4. Claude wybiera najlepsze 15–60 s fragmenty, pisze hook i tytuł zgodnie z zasadami kampanii.
   Bez klucza `ANTHROPIC_API_KEY` wybierane są po prostu najgłośniejsze momenty.
5. Pobiera tylko wybrane fragmenty wideo, tnie je do 1080x1920 z napisami i nickiem streamera.
6. Wrzuca klipy, opisy do wklejenia i `REVIEW.md` jako artefakt przebiegu (zakładka Actions → przebieg → Artifacts).

Publikacja i zgłoszenie w Content Rewards zostają ręczne: klip trzeba zgłosić w ciągu 30 minut od publikacji.

## Konfiguracja

- `Settings → Secrets and variables → Actions → New repository secret`: `ANTHROPIC_API_KEY`.
- Streamerów i zasady kampanii zmieniasz w `config/streamers.yaml`. Dodawaj tylko streamerów, na których klipy masz zgodę.
- Ręczne uruchomienie: Actions → Clips → Run workflow (opcjonalnie streamer i link do konkretnego VOD).

## Lokalnie

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest
python -m pytest -q
python -m clipper --streamer lacy --dry-run
```

Wymaga `ffmpeg`.
