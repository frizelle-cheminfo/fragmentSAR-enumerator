from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse        
from fastapi.requests import Request 
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED

FG = {           # 30 groups, * = attachment point
    "F":"*F", "Cl":"*Cl", "Br":"*Br",
    "Me":"*C", "Et":"*CC", "iPr":"*C(C)C", "tBu":"*C(C)(C)C",
    "CF3":"*C(F)(F)F", "OCF3":"*OC(F)(F)F",
    "OH":"*O", "OMe":"*OC", "OEt":"*OCC", "CF2H":"*C(F)FH",
    "NH2":"*N", "NMe2":"*N(C)C", "CONH2":"*C(=O)N", 
"SO2NH2":"*S(=O)(=O)N",
    "Acryl":"*C(=O)C=C", "ClAc":"*C(=O)CCl", "SulfonylF":"*S(=O)(=O)F",
    "Boro":"*B(O)O", "Isothio":"*N=C=S", "Ald":"*C=O",
    "Azide":"*N=[N+]=[N-]", "Alkyne":"*C#C", "Diazirine":"*C1N=N1",
    "Norborn":"*C1CCCC2CC1C2",
    "2‑Pyr":"*c1ncccc1", "4‑Pyr":"*c1ccncc1", "Imid":"*c1ncc[nH]1", 
"Thiaz":"*c1nccs1"
}

app = FastAPI(title="Fragment‑SAR enumerator")
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(Exception)
async def all_errors(request: Request, exc: Exception):
    """
    Catch *any* uncaught exception and turn it into
    JSON so the front‑end gets a clean 500 response.
    """
    return JSONResponse(
        {"detail": str(exc)},   # what went wrong
        status_code=500
    )
def desc(m):
    d = {
        "smiles": Chem.MolToSmiles(m, True),
        "mw": Descriptors.MolWt(m),
        "clogp": Descriptors.MolLogP(m),
        "hbd": Descriptors.NumHDonors(m),
        "hba": Descriptors.NumHAcceptors(m),
        "qed": QED.qed(m),
    }
    d["ro5"] = (d["mw"]>500)+(d["clogp"]>5)+(d["hbd"]>5)+(d["hba"]>10)
    return d

@app.post("/enumerate")
def run(smiles: str, groups: list[str] | None = None, limit: int = 200):
    core = Chem.MolFromSmiles(smiles)
    if core is None:
        raise HTTPException(400, "Bad SMILES")
    groups = groups or list(FG)

    patt_H = Chem.MolFromSmarts("[H]")
    products = {}
    for a in core.GetAtoms():
        if a.GetTotalNumHs():
            for tag in groups:
                fg = Chem.MolFromSmiles(FG[tag])
                reps = AllChem.ReplaceSubstructs(
                    core, patt_H, fg,
                    replacementConnectionPoint=0, replaceAll=False,
                    substructs=[(a.GetIdx(),)]
                )
                for p in reps:
                    Chem.SanitizeMol(p, catchErrors=True)
                    products[Chem.MolToSmiles(p, True)] = p
                    if len(products) >= limit: break
            if len(products) >= limit: break

    return {"rows": [desc(m) for m in products.values()]}

