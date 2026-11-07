# Attribution and retained notices

The translated ORB-SLAM3 algorithms and retained ORB vocabulary are accompanied
by the upstream GPL version 3 text in `LICENSE` and author attribution in
`NOTICE`. `BASELINE.json` records the exact local source hashes. There is no
ORB-SLAM3 native library in this package.

| Material | Notice retained |
| --- | --- |
| ORB extraction and 256-bit sampling pattern, originally Willow Garage/OpenCV | `licenses/ORB-WillowGarage-BSD.txt` |
| DBoW2 vocabulary transform, scoring and retrieval, Dorian Gálvez-López | `licenses/DBoW2.txt`, original bundled README retained separately |
| MLPnP, Steffen Urban/ANU, 2016 | `licenses/MLPnP-ANU-BSD.txt`, also in `orbslam3/pnp.py` |
| g2o LM/graph numerical procedures | `licenses/g2o-BSD.txt` |
| Sophus mathematical conventions | `licenses/Sophus-MIT.txt` |

The local DBoW2 checkout omits the `LICENSE.txt` referenced by its README.
The notice included here was obtained from the
[upstream DBoW2 license](https://raw.githubusercontent.com/dorian3d/DBoW2/master/LICENSE.txt).
It contains an author-notification condition for redistribution as well as
attribution and non-endorsement conditions; preserve its complete terms with
any redistribution. This private implementation work has not notified or
sent messages to any third party.

NumPy, SciPy, OpenCV, pytest and Matplotlib retain their own notices in the
installed distributions. They provide numerical/image primitives, testing,
and optional plotting; they do not hide an external SLAM implementation.
