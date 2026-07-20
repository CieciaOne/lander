# Postępy pracy magisterskiej

**Temat:** Analiza technik continual learning w modelach uczących się nowych informacji w locie.
**Domena testowa:** nawigacja łazika marsjańskiego (rocker-bogie, 6 kół) w symulacji MuJoCo.
**Autor:** Jakub Ciećka · stan na 7 czerwca 2026.

## Jak zmienił się zakres

Pierwotny pomysł był szerokim przeglądem continual learningu w sterowaniu robotami i zakładał trzy luźno opisane scenariusze (nawigacja typu Roomba, manipulacja ramieniem, lokomocja humanoida) oraz dwie–trzy metody do porównania. W trakcie pracy zakres się zawęził i pogłębił.

Zamiast trzech płytkich scenariuszy z różnych domen wybrałem jeden, ale realistyczny i trudny: łazik marsjański z pasywnym zawieszeniem rocker-bogie uczący się jeździć po kolejnych terenach. Pozwala to rzetelnie porównać metody CL na spójnym problemie, a nie rozdrabniać się na trzy środowiska. Liczba badanych technik wzrosła za to z dwóch–trzech do siedmiu.

|  | Pierwotny pomysł | Stan obecny |
|---|---|---|
| Domena | otwarta: Roomba, Fetch/Franka, Ant/Humanoid | jeden pogłębiony przypadek: łazik marsjański |
| Środowisko | gotowe modele MuJoCo/Gymnasium | własny model MJCF łazika, katalog terenów, heightmapy, backend GPU (MJX) |
| Metody CL | EWC/LwF + replay (2–3) | naive, replay, EWC, L2, MAS, distillation (LwF), hybryda EWC+replay |
| Oś trudności | zmiana layoutu / terenu | krzywa terenów z randomizacją i kotwicami przeciw zapominaniu |
| Bezpieczeństwo | brak | dodane później w planie, obecnie kandydat na dalsze prace |

## Co jest gotowe

Środowisko jest skończone. Mam własny model łazika w MJCF (pasywne zawieszenie rocker-bogie, 14 aktuatorów, grawitacja marsjańska), środowisko Gymnasium z dwuwymiarową akcją Ackermanna i czterdziestowymiarową obserwacją, deklaratywny katalog terenów (statyczne T1–T6 oraz randomizowane RT i RC) oraz proceduralne heightmapy dla terenu nierównego.

Zaimplementowane jest siedem technik CL pokrywających trzy klasy z taksonomii: regularizację (EWC, L2, MAS), distillation (LwF) i replay, plus hybrydę EWC z replayem. Wszystkie działają na wspólnym PPO.

Działa też infrastruktura eksperymentalna: klasy Mission i Runner trenują zadania po kolei i po każdej fazie liczą wyniki na wszystkich dotąd widzianych terenach, zapisując wyniki, checkpointy i wykresy (macierz retencji, krzywe, przeżywalność umiejętności). Jest backend MJX uruchamiający wiele łazików równolegle na GPU, obsługa wielu ziaren losowych oraz 15 plików testów (czas wykonania około 45 sekund).

Powstało 13 scenariuszy, od podstawowych testów zapominania, przez badanie wpływu kolejności zadań i rozmiaru bufora replay, po dwa kluczowe: `scenario_12_joint_training` jako baseline górnej granicy oraz `scenario_13_integrated_curriculum` jako obecnie najlepszy projekt krzywej uczenia.

Z dotychczasowych prób wynika jeden istotny wniosek metodyczny. Fazy ćwiczące pojedynczą umiejętność (na przykład „tylko przeszkody" po „tylko ścieżce") prowadzą do silnego zapominania, bo gradient nowego zadania niszczy cechy potrzebne staremu. Rozwiązaniem okazało się projektowanie krzywej tak, by każda faza po fazie wstępnej zawierała w każdym epizodzie zarówno przeszkody, jak i punkty trasy. Zadanie metody CL sprowadza się wtedy do utrzymania stopniowo poprawiających się umiejętności, a nie do ochrony jednej umiejętności w trakcie nauki zupełnie innej.

## W toku

Trwa strojenie hiperparametrów i wybór konfiguracji. Część uruchomień na scenariuszach 12 i 13 ma już zapisane wyniki dla niektórych metod i ziaren, ale brakuje kompletnej tabeli porównującej retencję wszystkich siedmiu metod na kilku ziarnach. Równolegle dostrajam samą krzywą uczenia (gęstość i odległości w fazach z przeszkodami).

## Do zrobienia

- Spisać formalnie przegląd technik CL (taksonomia i uzasadnienie wyboru metod) na rozdział teoretyczny. Materiał jest już w pierwotnym dokumencie i notatkach projektowych.
- Domknąć uruchomienia na wielu ziarnach i zestawić tabelę porównawczą oraz odpowiedź na pytanie badawcze: czy i o ile poszczególne metody ograniczają zapominanie względem zwykłego fine-tuningu i baseline'u joint.
- Podjąć decyzję o rozszerzeniach security i fusion. Nie było ich w pierwotnym pomyśle, w kodzie są obecnie zaślepkami i są naturalnym kandydatem na dalsze prace.
- Napisać raport końcowy z syntezą wyników i sekcją o ograniczeniach (symulacja, brak transferu sim-to-real, kwestie bezpieczeństwa).

## Najbliższe kroki

Priorytetem jest domknięcie uruchomień CL na scenariuszu 13 dla wszystkich siedmiu metod i ziaren 0–2 oraz policzenie macierzy retencji i zapominania względem baseline'u joint. Następnie spisanie przeglądu technik i sformułowanie odpowiedzi na pytanie badawcze. Decyzja o zakresie części security/fusion wpływa na harmonogram, więc warto ją podjąć wcześnie.
