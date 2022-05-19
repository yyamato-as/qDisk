# ver0; based on SourceList_noCam_sai.pdf
source_dict = {
    "L1489IRS": {
        "radec": "04h04m43.079964s 26d18m56.118528s", # from 2D Gaussian fit to SB+LB data with robust=1.0 
        "distance": 140,
        "PA": 69, # Sai et al. 2020
        "incl": 73, # Sai et al. 2020
        "v_sys": 7.3, # LSR systemic velocity in km/s; Sai et al. 2020
        "emission_extent": {"12CO": (-6.0, 22),
                            "13CO": (-0.20, 14),
                            "C18O": (1.6, 12.8),
                            "SO": (4.3, 10.7),
                            } # inspected on casaviewer by eye in robust=0.5 SB+LB images in km/s
    },
    "IRAS04169": {
        "radec": "04h19m58.449s  27d09m56.936s", # Takakuwa et al. 2018 (2D Gaussian fit)
    },
    "IRAS04302": {
        "radec": "04h33m16.49977s +22d53m20.225224s",
    },
    "Ced110IRS4": {
        "radec": "11h06m46.37687s -77d22m32.881218s",
    },
    "GSS30IRS3": {
        "radec": "16h26m21.72s -24d22m50.7s",
    },
    "OphIRS43": {
        "radec": "16h27m26.905457s -24d40m50.83194s",
    },
    "OphIRS63": {
        "radec": "16h31m35.70s -24d01m29.6s",
    },
    "IRS5N": {
        "radec": "19h01m48.479616s -36d57m15.38531s",
    },
    "IRS7B": {
        "radec": "19h01m56.419063s -36d57m28.67292s",
    },
    "IRAS32": {
        "radec": "19h02m58.72279s -37d07m37.38115s",
    },
    "IRAS04166+2706": {
        "radec": "04h19m42.50s  27d13m36.0s",
    },
    "L1527IRS": {
        "radec": "04h39m53.91s  26d03m09.8s",
    },
    "BHR71_IRS1": {
        "radec": "12h01m36.474422s -65d08m49.35978s", 
    },
    "BHR71_IRS2": {
        "radec": "12h01m34.01015s -65d08m48.0695s",
    },
    "IRAS15398": {
        "radec": "15h43m02.23327s -034d09m06.943163s",
    },
    "IRAS16253": {
        "radec": "16h28m21.615631s -24d36m24.32560s",
    },
    "CB68": {
        "radec": "16h57m19.642s -16d09m24.018s",
    },
}
